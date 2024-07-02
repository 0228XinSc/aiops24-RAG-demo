import asyncio
import json
from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.legacy.llms import OpenAILike as OpenAI
from llama_index.core.retrievers import BM25Retriever
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval
from llama_index.core.postprocessor import SentenceTransformerRerank
import argparse
from custom.template import QA_TEMPLATES

all_emds = {
    'BAAI': ('BAAI/bge-small-zh-v1.5', 512),
    'BAAI-L': ('BAAI/bge-large-zh-v1.5', 1024),
    'BAAI-B': ('BAAI/bge-large-zh-v1.5', 768),
    'GTE-L': ('thenlper/gte-large-zh', 1024),
    'GTE-L1': ('ICC/gte-large-zh', 1024),
    'GTE-B': ('thenlper/gte-base-zh', 768),
    'GTE-B': ('thenlper/gte-base-zh', 768),
    'OpenAI-L': ('text-embedding-3-large', 1024),
    'M3E': ('M3E/m3e-base', 768),
    'BCE-B': ('maidalun1020/bce-embedding-base_v1', )
    # 'SENSE-L-v2': {'name': 'sensenova/piccolo-large-zh-v2', 'dim': 1792},
    # 'SENSE-L': {'name': 'sensenova/piccolo-large-zh', 'dim': 1024},
    # 'SENSE-B': {'name': 'sensenova/piccolo-base-zh', 'dim': 768},
}
all_reranker = {
    'bge-m3': 'BAAI/bge-reranker-v2-m3',
    'bge-l': 'BAAI/bge-reranker-large',
    'bce-b': 'maidalun1020/bce-reranker-base_v1',
}
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-emd', type=str, default='BAAI')
    parser.add_argument('-reranker', type=str, default='bge-m3')
    parser.add_argument('-r_top_k', type=int, default=30)
    parser.add_argument('-qat_idx', type=int, default=0)
    parser.add_argument('--use_reranker', action='store_true')
    parser.add_argument('-reranker_top_k', type=int, default=11)
    parser.add_argument('-data', type=str, default='data')
    parser.add_argument('-q', type=str, default='question.jsonl')
    args = parser.parse_args()
    args.use_reranker = True
    # config = dotenv_values(".env")
    config = {'COLLECTION_NAME':'aiops24',
              'GLM_KEY': 'f766070bdbd498c3b4c7c2e94df4f8e3.b4vtXk7XyeIUfmu3',
              'VECTOR_SIZE': all_emds[args.emd][1]}
    config["COLLECTION_NAME"] += '_' + args.emd
    prefix = ''
    if args.data != 'data':
        prefix = 'merge'
    reranker = None
    if args.use_reranker:
        reranker = SentenceTransformerRerank(top_n=args.reranker_top_k, model=all_reranker[args.reranker])
        print('Reranker加载完毕')
    qa_template = QA_TEMPLATES[args.qat_idx]
    # 初始化 LLM 嵌入模型 和 Reranker
    llm = OpenAI(
        api_key=config["GLM_KEY"],
        model="glm-4",
        api_base="https://open.bigmodel.cn/api/paas/v4/",
        is_chat_model=True,
    )
    print('LLM加载完毕')
    if 'OpenAI' in args.emd:
        embeding = OpenAIEmbedding(
            model=all_emds[args.emd][0],
            embed_batch_size=512,
        )
    else:
        embeding = HuggingFaceEmbedding(
            model_name=all_emds[args.emd][0],
            cache_folder="./",
            embed_batch_size=512,
        )
    print('Embedding加载完毕')
    Settings.embed_model = embeding

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)
    print('store build完毕')

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )
    print('collection_info获取完毕')
    print(collection_info.points_count)

    if collection_info.points_count == 0:
        data = read_data(args.data, num_workers=8)
        print('数据读取完毕')
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
        print('pipeline构建完毕')
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        print('update_collection完毕')
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    retriever = QdrantRetriever(vector_store, embeding, similarity_top_k=args.r_top_k)

    # 将结果保存为 JSON 文件
    with open('search_results.json', 'w', encoding='utf-8') as f:
        json.dump(retriever, f, ensure_ascii=False, indent=4)
    queries = read_jsonl(args.q)

    # 生成答案
    print("Start generating answers...")

    results = []
    for idx, query in enumerate(tqdm(queries, total=len(queries))):
        try:
            result = await generation_with_knowledge_retrieval(
                query["query"], retriever, llm, qa_template=qa_template, reranker=reranker,
            )
            results.append(result)
        except:
            print(idx)

    # 处理结果
    save_answers(queries, results, f"{prefix}_{args.reranker}_{args.reranker_top_k}_{args.qat_idx}_{args.emd}_{args.r_top_k}_submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
