import "dotenv/config";
import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

//获取网页内容
const cheerioLoader = new CheerioWebBaseLoader(
  "https://juejin.cn/post/7683457864929329204",
  {
    selector: ".main-area p",
  },
);

const documents = await cheerioLoader.load();

// console.log(documents);

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,
  chunkOverlap: 50,
  separators: ["。", "！", "？"],
});

const splitDocuments = await textSplitter.splitDocuments(documents);

console.log(splitDocuments);
