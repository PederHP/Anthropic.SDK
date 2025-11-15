using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Anthropic.SDK.Common;
using Microsoft.Extensions.AI;

namespace Anthropic.SDK.Messaging;

public partial class MessagesEndpoint : IChatClient
{
    private ChatClientMetadata _metadata;

    /// <inheritdoc />
    async Task<ChatResponse> IChatClient.GetResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions options, CancellationToken cancellationToken)
    {
        MessageResponse response = await this.GetClaudeMessageAsync(ChatClientHelper.CreateMessageParameters(this, messages, options), cancellationToken);

        ChatMessage message = new(ChatRole.Assistant, ChatClientHelper.ProcessResponseContent(response));

        if (response.StopSequence is not null)
        {
            (message.AdditionalProperties ??= [])[nameof(response.StopSequence)] = response.StopSequence;
        }

        if (response.RateLimits is { } rateLimits)
        {
            Dictionary<string, object> d = new();
            (message.AdditionalProperties ??= [])[nameof(response.RateLimits)] = d;

            if (rateLimits.RequestsLimit is { } requestLimit)
            {
                d[nameof(rateLimits.RequestsLimit)] = requestLimit;
            }

            if (rateLimits.RequestsRemaining is { } requestsRemaining)
            {
                d[nameof(rateLimits.RequestsRemaining)] = requestsRemaining;
            }

            if (rateLimits.RequestsReset is { } requestsReset)
            {
                d[nameof(rateLimits.RequestsReset)] = requestsReset;
            }

            if (rateLimits.RetryAfter is { } retryAfter)
            {
                d[nameof(rateLimits.RetryAfter)] = retryAfter;
            }

            if (rateLimits.TokensLimit is { } tokensLimit)
            {
                d[nameof(rateLimits.TokensLimit)] = tokensLimit;
            }

            if (rateLimits.TokensRemaining is { } tokensRemaining)
            {
                d[nameof(rateLimits.TokensRemaining)] = tokensRemaining;
            }

            if (rateLimits.TokensReset is { } tokensReset)
            {
                d[nameof(rateLimits.TokensReset)] = tokensReset;
            }
        }

        return new(message)
        {
            ResponseId = response.Id,
            FinishReason = response.StopReason switch
            {
                "max_tokens" => ChatFinishReason.Length,
                _ => ChatFinishReason.Stop,
            },
            ModelId = response.Model,
            RawRepresentation = response,
            Usage = response.Usage is { } usage ? ChatClientHelper.CreateUsageDetails(usage) : null
        };
    }

    /// <inheritdoc />
    async IAsyncEnumerable<ChatResponseUpdate> IChatClient.GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var thinking = string.Empty;
        await foreach (MessageResponse response in StreamClaudeMessageAsync(ChatClientHelper.CreateMessageParameters(this, messages, options), cancellationToken))
        {
            var update = new ChatResponseUpdate
            {
                ResponseId = response.Id,
                ModelId = response.Model,
                RawRepresentation = response,
                Role = ChatRole.Assistant
            };

            if (!string.IsNullOrEmpty(response.ContentBlock?.Data))
            {
                update.Contents.Add(new TextReasoningContent(null) { ProtectedData = response.ContentBlock.Data });
            }

            // Handle server tool content blocks during streaming
            if (response.ContentBlock is not null)
            {
                // Handle server_tool_use content blocks
                if (response.ContentBlock.Type == "server_tool_use" && 
                    !string.IsNullOrEmpty(response.ContentBlock.Id) && 
                    !string.IsNullOrEmpty(response.ContentBlock.Name))
                {
                    // For server tools, we emit a FunctionCallContent during streaming
                    update.Contents.Add(new FunctionCallContent(
                        response.ContentBlock.Id,
                        response.ContentBlock.Name,
                        new Dictionary<string, object>()));
                }
                
                // Handle tool result content blocks (web_search_tool_result, bash_code_execution_tool_result, etc.)
                if ((response.ContentBlock.Type == "web_search_tool_result" ||
                     response.ContentBlock.Type == "bash_code_execution_tool_result" ||
                     response.ContentBlock.Type == "text_editor_code_execution_tool_result" ||
                     response.ContentBlock.Type == "mcp_tool_result") &&
                    !string.IsNullOrEmpty(response.ContentBlock.ToolUseId) &&
                    response.ContentBlock.Content is not null)
                {
                    // Convert content to text representation
                    var resultText = new System.Text.StringBuilder();
                    foreach (var content in response.ContentBlock.Content)
                    {
                        if (content is TextContent tc)
                        {
                            resultText.AppendLine(tc.Text);
                        }
                        else if (content is WebSearchResultContent wsrc)
                        {
                            resultText.AppendLine($"Title: {wsrc.Title}");
                            resultText.AppendLine($"URL: {wsrc.Url}");
                            if (!string.IsNullOrEmpty(wsrc.PageAge))
                                resultText.AppendLine($"Page Age: {wsrc.PageAge}");
                            resultText.AppendLine();
                        }
                    }
                    
                    // Emit a FunctionResultContent for the server tool result
                    update.Contents.Add(new FunctionResultContent(
                        response.ContentBlock.ToolUseId,
                        resultText.ToString())
                    {
                        RawRepresentation = response.ContentBlock
                    });
                }
            }
            
            if (response.StreamStartMessage?.Usage is {} startStreamMessageUsage)
            {
                update.Contents.Add(new UsageContent(ChatClientHelper.CreateUsageDetails(startStreamMessageUsage)));
            }
            
            if (response.Delta is not null)
            {
                if (!string.IsNullOrEmpty(response.Delta.Text))
                {
                    update.Contents.Add(new Microsoft.Extensions.AI.TextContent(response.Delta.Text));
                }

                if (!string.IsNullOrEmpty(response.Delta.Thinking))
                {
                    thinking += response.Delta.Thinking;
                }

                if (!string.IsNullOrEmpty(response.Delta.Signature))
                {
                    update.Contents.Add(new TextReasoningContent(thinking)
                    {
                        ProtectedData = response.Delta.Signature,
                    });
                }

                if (response.Delta?.StopReason is string stopReason)
                {
                    update.FinishReason = response.Delta.StopReason switch
                    {
                        "max_tokens" => ChatFinishReason.Length,
                        _ => ChatFinishReason.Stop,
                    };
                }

                if (response.Usage is { } usage)
                {
                    update.Contents.Add(new UsageContent(ChatClientHelper.CreateUsageDetails(usage)));
                }
            }

            if (response.ToolCalls is { Count: > 0 })
            {
                foreach (var f in response.ToolCalls)
                {
                    update.Contents.Add(new FunctionCallContent(f.Id, f.Name,
                        !string.IsNullOrEmpty(f.Arguments.ToString())
                            ? JsonSerializer.Deserialize<Dictionary<string, object>>(f.Arguments.ToString())
                            : new Dictionary<string, object>()));
                }
                
            }

            yield return update;
        }
    }

    /// <inheritdoc />
    void IDisposable.Dispose() { }

    /// <inheritdoc />
    object IChatClient.GetService(Type serviceType, object serviceKey) =>
        serviceKey is not null ? null :
        serviceType == typeof(ChatClientMetadata) ? (_metadata ??= new(nameof(AnthropicClient), new Uri(Url))) :
        serviceType?.IsInstanceOfType(this) is true ? this :
        null;
}
