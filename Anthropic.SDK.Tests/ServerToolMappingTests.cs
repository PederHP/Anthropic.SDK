using Anthropic.SDK.Messaging;
using Microsoft.Extensions.AI;
using System.Collections.Generic;
using System.Linq;

namespace Anthropic.SDK.Tests
{
    [TestClass]
    public class ServerToolMappingTests
    {
        [TestMethod]
        public void TestServerToolUseContentMapping()
        {
            // Arrange - Create a MessageResponse with server_tool_use content
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new ServerToolUseContent
                    {
                        Id = "toolu_123",
                        Name = "web_search",
                        Input = new ServerToolInput
                        {
                            Query = "weather in San Francisco"
                        }
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify ServerToolUseContent is mapped to FunctionCallContent
            Assert.AreEqual(1, aiContents.Count);
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionCallContent));
            
            var functionCall = aiContents[0] as FunctionCallContent;
            Assert.AreEqual("toolu_123", functionCall.CallId);
            Assert.AreEqual("web_search", functionCall.Name);
            Assert.IsNotNull(functionCall.Arguments);
            Assert.IsTrue(functionCall.Arguments.ContainsKey("query"));
            Assert.AreEqual("weather in San Francisco", functionCall.Arguments["query"]);
        }

        [TestMethod]
        public void TestWebSearchToolResultContentMapping()
        {
            // Arrange - Create a MessageResponse with web_search_tool_result content
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new WebSearchToolResultContent
                    {
                        ToolUseId = "toolu_123",
                        Content = new List<ContentBase>
                        {
                            new WebSearchResultContent
                            {
                                Title = "San Francisco Weather",
                                Url = "https://weather.com/sf",
                                PageAge = "1 day ago"
                            }
                        }
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify WebSearchToolResultContent is mapped to FunctionResultContent
            Assert.AreEqual(1, aiContents.Count);
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionResultContent));
            
            var functionResult = aiContents[0] as FunctionResultContent;
            Assert.AreEqual("toolu_123", functionResult.CallId);
            Assert.IsNotNull(functionResult.Result);
            
            var resultText = functionResult.Result.ToString();
            Assert.IsTrue(resultText.Contains("San Francisco Weather"));
            Assert.IsTrue(resultText.Contains("https://weather.com/sf"));
            Assert.IsTrue(resultText.Contains("1 day ago"));
        }

        [TestMethod]
        public void TestServerToolUseAndResultTogether()
        {
            // Arrange - Create a MessageResponse with both server_tool_use and result
            // This simulates what happens with server-side tools
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new ServerToolUseContent
                    {
                        Id = "toolu_123",
                        Name = "web_search",
                        Input = new ServerToolInput
                        {
                            Query = "weather in San Francisco"
                        }
                    },
                    new WebSearchToolResultContent
                    {
                        ToolUseId = "toolu_123",
                        Content = new List<ContentBase>
                        {
                            new WebSearchResultContent
                            {
                                Title = "San Francisco Weather",
                                Url = "https://weather.com/sf"
                            }
                        }
                    },
                    new Anthropic.SDK.Messaging.TextContent
                    {
                        Text = "Based on the search results, the weather in San Francisco is..."
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify we get FunctionCallContent, FunctionResultContent, and TextContent
            Assert.AreEqual(3, aiContents.Count);
            
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionCallContent));
            var functionCall = aiContents[0] as FunctionCallContent;
            Assert.AreEqual("toolu_123", functionCall.CallId);
            Assert.AreEqual("web_search", functionCall.Name);
            
            Assert.IsInstanceOfType(aiContents[1], typeof(FunctionResultContent));
            var functionResult = aiContents[1] as FunctionResultContent;
            Assert.AreEqual("toolu_123", functionResult.CallId);
            
            Assert.IsInstanceOfType(aiContents[2], typeof(Microsoft.Extensions.AI.TextContent));
            var textContent = aiContents[2] as Microsoft.Extensions.AI.TextContent;
            Assert.IsTrue(textContent.Text.Contains("Based on the search results"));
        }

        [TestMethod]
        public void TestBashCodeExecutionToolResultMapping()
        {
            // Arrange - Create a MessageResponse with bash_code_execution_tool_result
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new BashCodeExecutionToolResultContent
                    {
                        ToolUseId = "toolu_456",
                        Content = new BashCodeExecutionResultContent
                        {
                            Stdout = "Hello World",
                            Stderr = "",
                            ReturnCode = 0
                        }
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify mapping to FunctionResultContent
            Assert.AreEqual(1, aiContents.Count);
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionResultContent));
            
            var functionResult = aiContents[0] as FunctionResultContent;
            Assert.AreEqual("toolu_456", functionResult.CallId);
            
            var resultText = functionResult.Result.ToString();
            Assert.IsTrue(resultText.Contains("stdout: Hello World"));
            Assert.IsTrue(resultText.Contains("return_code: 0"));
        }

        [TestMethod]
        public void TestMCPToolUseContentMapping()
        {
            // Arrange - Create a MessageResponse with mcp_tool_use content
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new MCPToolUseContent
                    {
                        Id = "toolu_789",
                        Name = "get_repo_info",
                        ServerName = "DeepWiki",
                        Input = System.Text.Json.Nodes.JsonNode.Parse("{\"repo\":\"anthropic/sdk\"}")
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify MCPToolUseContent is mapped to FunctionCallContent
            Assert.AreEqual(1, aiContents.Count);
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionCallContent));
            
            var functionCall = aiContents[0] as FunctionCallContent;
            Assert.AreEqual("toolu_789", functionCall.CallId);
            Assert.AreEqual("get_repo_info", functionCall.Name);
        }

        [TestMethod]
        public void TestMCPToolResultContentMapping()
        {
            // Arrange - Create a MessageResponse with mcp_tool_result content
            var messageResponse = new MessageResponse
            {
                Id = "msg_123",
                Model = "claude-3-sonnet",
                Content = new List<ContentBase>
                {
                    new MCPToolResultContent
                    {
                        ToolUseId = "toolu_789",
                        Content = new List<ContentBase>
                        {
                            new Anthropic.SDK.Messaging.TextContent
                            {
                                Text = "Repository information: Anthropic SDK"
                            }
                        }
                    }
                }
            };

            // Act - Process the response content
            var aiContents = ChatClientHelper.ProcessResponseContent(messageResponse);

            // Assert - Verify MCPToolResultContent is mapped to FunctionResultContent
            Assert.AreEqual(1, aiContents.Count);
            Assert.IsInstanceOfType(aiContents[0], typeof(FunctionResultContent));
            
            var functionResult = aiContents[0] as FunctionResultContent;
            Assert.AreEqual("toolu_789", functionResult.CallId);
            
            var resultText = functionResult.Result.ToString();
            Assert.IsTrue(resultText.Contains("Repository information"));
        }
    }
}
