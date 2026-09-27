import "dotenv/config";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const model = new ChatOpenAI({
  temperature: 0,
  model: process.env.MODEL_NAME,
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.EMBEDDLINGS_API_KEY,
  model: process.env.EMBEDDINGS_MODEL_NAME,
  configuration: {
    baseURL: process.env.EMBEDDINGS_BASE_URL,
  },
});

//获取网页内容
const cheerioLoader = new CheerioWebBaseLoader(
  "https://juejin.cn/post/7683457864929329204",
  {
    selector: ".main-area p",
  },
);

const documents = await cheerioLoader.load();

console.log(`Total characters：${documents[0].pageContent.length}`);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,
  chunkOverlap: 50,
  separators: ["。", "！", "？"],
});

const splitDocuments = await textSplitter.splitDocuments(documents);

console.log(`文档分割完成，共${splitDocuments.length}个分块`);
console.log("正在创建向量存储...");

const vectorStore = await MemoryVectorStore.fromDocuments(
  documents,
  embeddings,
);
console.log("向量存储创建完成\n");

const reterever = vectorStore.asRetriever({ k: 2 });

const questions = ["飞鱼为什么不同意做Agent开发？"];

for (const question of questions) {
  console.log("=".repeat(80));
  console.log("问题：", question);
  console.log("=".repeat(80));

  //使用retriever 获取文档
  const retrievedDocs = await reterever.invoke(question);

  //使用similaritySearchWithScore获取相似度评分
  const scoredResults = await vectorStore.similaritySearchWithScore(
    question,
    3,
  );

  //打印用到的文档和相似度评分
  console.log("\n【检索到文档及相似度评分】");
  retrievedDocs.forEach((doc, i) => {
    const scoredResult = scoredResults.find(
      ([scoreDoc]) => scoreDoc.pageContent === doc.pageContent,
    );
    const score = scoredResult ? scoredResult[1] : null;
    const similarity = score !== null ? (1 - score).toFixed(4) : "N/A";
    console.log(`\n文档${i + 1}余弦相似度：${similarity}`);
    console.log(`内容：${doc.pageContent}`);
    console.log(`元数据：`, doc.metadata);
  });

  //构建prompt
  const context = retrievedDocs
    .map((doc, i) => `[片段${i + 1}]\n${doc.pageContent}`)
    .join("\n\n——————\n\n");
  const prompt = `你是一个文章辅助阅读助手，根据文章内容来解答。
文章内容:${context}

问题：${question}

你的回答：
`;
  console.log("\n【AI回答】");
  const response = await model.invoke(prompt);
  console.log(response.content);
  console.log("\n");
}
