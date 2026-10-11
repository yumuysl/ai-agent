// import "dotenv/config";
// import { ChatOpenAI } from "@langchain/openai";
import { InMemoryChatMessageHistory } from "@langchain/core/chat_history";
import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  trimMessages,
} from "@langchain/core/messages";
import { getEncoding } from "js-tiktoken";

// =========== 1. 按消息数量截断 ===================
async function messageCountTruncation() {
  const history = new InMemoryChatMessageHistory();
  const maxMessages = 4;

  const messages = [
    {
      type: "human",
      content: "我叫张三",
    },
    {
      type: "ai",
      content: "你好张三，很高兴认识你!",
    },
    {
      type: "human",
      content: "我今年25岁",
    },
    {
      type: "ai",
      content: "25岁正是青春年华，有什么问题可以帮助你的吗？",
    },
    {
      type: "human",
      content: "我喜欢编程",
    },
    {
      type: "ai",
      content: "编程很有趣，你主要用什么语言？",
    },
    {
      type: "human",
      content: "我住在北京",
    },
    {
      type: "ai",
      content: "北京是个很棒的城市!",
    },
    {
      type: "human",
      content: "我的职业是软件工程师",
    },
    {
      type: "ai",
      content: "软件工程师是个很有前景的职业！",
    },
  ];

  //添加所有消息
  for (const msg of messages) {
    if (msg.type == "human") {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  let allMessages = await history.getMessages();

  //按消息数量截断：保留最近maxMessages条消息
  const trimmedMessages = allMessages.slice(-maxMessages);
  console.log(`保留消息数量：${trimmedMessages.length}`);
  console.log(
    "保留的消息：",
    trimmedMessages
      .map((m) => `${m.constructor.name}: ${m.content}`)
      .join("\n "),
  );
}

//计算消息数组的总token数量
function countTokens(messages, encoder) {
  let total = 0;
  for (const msg of messages) {
    const content =
      typeof msg.content === "string"
        ? msg.content
        : JSON.stringify(msg.content);
    total += encoder.encode(content).length;
  }

  return total;
}

// =========== 2. 按token数量截断 ===================
async function tokenCountTruncation() {
  const history = new InMemoryChatMessageHistory();
  const maxTokens = 100;

  const enc = getEncoding("cl100k_base");

  const messages = [
    {
      type: "human",
      content: "我叫李四",
    },
    {
      type: "ai",
      content: "你好李四，很高兴认识你!",
    },
    {
      type: "human",
      content: "我是一名设计师",
    },
    {
      type: "ai",
      content: "设计师是个很有创造力的职业！你主要做什么类型的设计？",
    },
    {
      type: "human",
      content: "我喜欢艺术和音乐",
    },
    {
      type: "ai",
      content: "艺术和音乐都说很好的爱好，它们能激发创作灵感",
    },
    {
      type: "human",
      content: "我擅长UI/UX设计",
    },
    {
      type: "ai",
      content: "UI/UX设计非常重要，好的用户体验能让产品更成功!",
    },
  ];

  //添加所有消息
  for (const msg of messages) {
    if (msg.type === "human") {
      await history.addMessage(new HumanMessage(msg.content));
    } else {
      await history.addMessage(new AIMessage(msg.content));
    }
  }

  let allMessages = await history.getMessages();

  //使用trimMessages API: 使用js-tiktoken 计算token数量
  const trimmedMessages = await trimMessages(allMessages, {
    maxTokens: maxTokens,
    tokenCounter: async (msgs) => countTokens(msgs, enc),
    strategy: "last",
  });

  //计算实际token数用于显示
  const totalTokens = countTokens(trimmedMessages, enc);
  console.log(`总token数：${totalTokens}/${maxTokens}`);
  console.log(`保留消息数量：${trimmedMessages.length}`);
  console.log(
    "保留的消息",
    trimmedMessages
      .map((m) => {
        const content =
          typeof m.content === "string" ? m.content : JSON.stringify(m.content);
        const tokens = enc.encode(content).length;
        return `${m.constructor.name} (${tokens} tokens): ${content}`;
      })
      .join(`\n `),
  );
}

async function runAll() {
  await messageCountTruncation();
  await tokenCountTruncation();
}

runAll().catch(console.error);
