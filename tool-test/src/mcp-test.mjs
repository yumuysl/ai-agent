import "dotenv/config";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { ChatOpenAI } from "@langchain/openai";
import chalk from "chalk";
import {
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";

const model = new ChatOpenAI({
  model: process.env.MODEL_NAME || "deepseek-v4-flash",
  apiKey: process.env.OPENAI_API_KEY,
  configuration: {
    baseURL: process.env.OPENAI_BASE_URL,
  },
});

const mcpClient = new MultiServerMCPClient({
  mcpServers: {
    "my-mcp-server": {
      command: "node",
      args: ["/home/yumumu/Projects/ai/tool-test/src/my-mcp-server.mjs"],
    },
    filesystem: {
      command: "npx",
      args: [
        "-y",
        "@modelcontextprotocol/server-filesystem@latest",
        ...(process.env.ALLOWD_PATHS.split(",") || ""),
      ],
    },
    "amap-maps-streamableHTTP": {
      url: "https://mcp.amap.com/mcp?key=" + process.env.AMAP_MAPS_API_KEY,
    },
    "chrome-devtools": {
      command: "npx",
      args: ["-y", "chrome-devtools-mcp@latest"],
    },
  },
});

const tools = await mcpClient.getTools();
const modelWithTools = model.bindTools(tools);

// const res = await mcpClient.listResources();
// // console.log(res);
//
// let resourceContent = "";
// for (const [serverName, resources] of Object.entries(res)) {
//   for (const resource of resources) {
//     // console.log(resource);
//     const content = await mcpClient.readResource(serverName, resource.uri);
//     resourceContent += content[0].text;
//   }
// }

async function runAgentWithTools(query, maxIterations = 30) {
  const messages = [
    new SystemMessage(`￼
     - 每一轮对话中，最多只能同时发起3个maps_工具调用;
     - 如果需要查询超过3个地点，必须分批执行，每批最多3个，等上一批全部返回结果后，再发起下一批；
     - 严谨在同一轮中一次性发起超过3个maps_工具调用
     `),
    new HumanMessage(query),
  ];

  for (let i = 0; i < maxIterations; i++) {
    console.log(chalk.bgGreen(`⏳ 正在等待 AI 思考...`));

    const response = await modelWithTools.invoke(messages);
    messages.push(response);

    if (!response || response.tool_calls.length === 0) {
      console.log(`\n✨ AI 最终回复:\n${response.content}\n`);
      return response.content;
    }

    console.log(
      chalk.bgGreen(`🔍 检测到 ${response.tool_calls.length} 个工具调用`),
    );
    console.log(
      chalk.bgBlue(
        `🔍 工具调用: ${response.tool_calls.map((t) => t.name).join(", ")}`,
      ),
    );

    for (const toolCall of response.tool_calls) {
      const foundTool = tools.find((t) => t.name === toolCall.name);
      if (foundTool) {
        const toolResult = await foundTool.invoke(toolCall.args);

        //确保content是字符串类型
        let contentStr;
        if (typeof toolResult === "string") {
          contentStr = toolResult;
        } else if (toolResult && toolResult.text) {
          //部分工具返回结果不是字符串，有text字段
          contentStr = toolResult.text;
        }

        messages.push(
          new ToolMessage({
            content: contentStr,
            tool_call_id: toolCall.id,
          }),
        );
      }
    }
  }

  return messages[messages.length - 1].content;
}

const case1 =
  "西安北站附件的酒店，最近的3个酒店，拿到酒店图片，打开浏览器，展示每个酒店的图片，每个 tab 一个 url 展示，并且在把那个页面标题改为酒店名";
const case2 =
  "西安北站附近的5个酒店，以及去的路线，路线规划生成文档保存到 /home/yumumu/Projects/ai/tool-test/tesfiles/ 的一个 md 文件";
await runAgentWithTools(case2);

await mcpClient.close();
