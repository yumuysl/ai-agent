import "dotenv/config";
import { parse } from "path";
import {
  MilvusClient,
  DataType,
  MetricType,
  IndexType,
} from "@zilliz/milvus2-sdk-node";
import { OpenAIEmbeddings } from "@langchain/openai";
import { EPubLoader } from "@langchain/community/document_loaders/fs/epub";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const COLLECTION_NAME = "ebook_collection";
const VECTOR_DIM = 1024;
const CHUNK_SIZE = 500;
// const EPUB_FILE = "./天龙八部.epub";
const EPUB_FILE = "/home/yumumu/Documents/books/《穆斯林的葬礼》---霍达.epub";

// 从文件名提取书名（去掉扩展名）
const BOOK_NAME = parse(EPUB_FILE).name;

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDLINGS_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: process.env.EMBEDDINGS_BASE_URL,
  },
  // dimensions: VECTOR_DIM,    //有些接口默认1024不接收dimensions参数
});

const client = new MilvusClient({
  address: "localhost:19530",
});

async function getEmbedding(text) {
  // console.log("getEmbedding：", text);
  const result = await embeddings.embedQuery(text);
  return result;
}

/**
 * 创建或获取集合
 */
async function ensureCollection(bookId) {
  try {
    //检查集合是否存在
    const hasCollection = await client.hasCollection({
      collection_name: COLLECTION_NAME,
    });

    if (!hasCollection.value) {
      console.log("创建集合...");
      await client.createCollection({
        collection_name: COLLECTION_NAME,
        fields: [
          {
            name: "id",
            data_type: DataType.VarChar,
            max_length: 100,
            is_primary_key: true,
          },
          {
            name: "book_id",
            data_type: DataType.VarChar,
            max_length: 100,
          },
          {
            name: "book_name",
            data_type: DataType.VarChar,
            max_length: 200,
          },
          {
            name: "chapter_num",
            data_type: DataType.Int32,
          },
          {
            name: "index",
            data_type: DataType.Int32,
          },
          {
            name: "content",
            data_type: DataType.VarChar,
            max_length: 5000,
          },
          {
            name: "vector",
            data_type: DataType.FloatVector,
            dim: VECTOR_DIM,
          },
        ],
      });

      console.log("集合创建成功");

      //创建索引
      console.log("创建索引...");
      await client.createIndex({
        collection_name: COLLECTION_NAME,
        field_name: "vector",
        index_type: IndexType.IVF_FLAT,
        metric_type: MetricType.COSINE,
        params: { nlist: 1024 },
      });

      console.log("创建索引成功");

      //确保集合已加载
      try {
        await client.loadCollection({ collection_name: COLLECTION_NAME });
        console.log("集合已加载");
      } catch (error) {
        console.log("集合已处于加载状态");
      }
    }
  } catch (error) {
    console.log("创建集合时出错：", error.message);
    throw error;
  }
}

/**
 * 将文档块批量插入到milvus（流式处理）
 */
async function insertChunksBatch(chunks, bookId, chapterNum) {
  try {
    console.log("chunks长度:", chunks.length);
    if (chunks.length === 0) {
      return 0;
    }

    //为每个文档生成向量并构建插入数据
    const insertData = await Promise.all(
      chunks.map(async (chunk, chunkIndex) => {
        const vector = await getEmbedding(chunk);
        //手动生成 ID: book_id_chapterNum_index
        return {
          id: `${bookId}_${chapterNum}_${chunkIndex}`,
          book_id: bookId,
          book_name: BOOK_NAME,
          chapter_num: chapterNum,
          index: chunkIndex,
          content: chunk,
          vector: vector,
        };
      }),
    );

    //批量插入到Milvus
    console.log("要插入数据的长度:", insertData.length);
    const insertResult = await client.insert({
      collection_name: COLLECTION_NAME,
      data: insertData,
    });

    console.log("插入返回结果状态：", insertResult.status);

    return Number(insertResult.insert_cnt) || 0;
  } catch (error) {
    console.error(`插入章节${chapterNum}的数据时出错：`, error.message);
    console.error("错误详情：", error);
    throw error;
  }
}

/**
 * 加载EPUB文件并进行流式处理（边处理边插入）
 */
async function loadAndProcessEPubStreaming(bookId) {
  try {
    console.log(`\n开始加载EPUB文件：${EPUB_FILE}`);

    //使用EPubLoader加载文件，按章节拆分
    const loader = new EPubLoader(EPUB_FILE, {
      splitChapters: true,
    });

    const documents = await loader.load();
    console.log(`加载完成，共${documents.length}个章节\n`);

    //创建文本拆分器，拆分到500个字符
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE,
      chunkOverlap: 50,
    });

    let totalInserted = 0;

    //遍历每个章节，进行第二次拆分并立即插入
    for (
      let chapterIndex = 0;
      chapterIndex < documents.length;
      chapterIndex++
    ) {
      const chapter = documents[chapterIndex];
      const chapterContent = chapter.pageContent;

      console.log(`处理第${chapterIndex + 1}/${documents.length}章...`);

      //使用splitter进行二次拆分
      const chunks = await textSplitter.splitText(chapterContent);

      console.log(`   拆分为${chunks.length}个片段`);

      if (chunks.length === 0) {
        console.log("  跳过空章节\n");
        continue;
      }

      console.log(` 生成向量并插入中...`);

      //立即生成向量并插入该章节的所有片段
      const insertedCount = await insertChunksBatch(
        chunks,
        bookId,
        chapterIndex + 1,
      );
      totalInserted += insertedCount;
      console.log(` 已插入${insertedCount}条记录（累记${totalInserted}）\n`);
    }

    console.log(`\n共插入${totalInserted}条记录`);
    return totalInserted;
  } catch (error) {
    console.error(`加载EPUB文件时出错：`, error.message);
    throw error;
  }
}

//主函数
async function main() {
  try {
    console.log("=".repeat(80));
    console.log("电子书处理程序");
    console.log("=".repeat(80));

    //连接Milvus
    console.log("\n连接Milvus...");
    await client.connectPromise;
    console.log("已连接\n");

    //设置book_id
    const bookId = 1;

    //确保集合存在
    await ensureCollection(bookId);

    //加载和处理EPUB文件（流式处理，边处理边插入）
    await loadAndProcessEPubStreaming(bookId);

    console.log("=".repeat(80));
    console.log("处理完成");
    console.log("=".repeat(80));
  } catch (error) {
    console.log("\n错误：", error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
