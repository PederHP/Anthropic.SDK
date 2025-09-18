using System;
using Microsoft.Extensions.AI;
using Anthropic.SDK.Messaging;

namespace Anthropic.SDK.Extensions
{
    /// <summary>
    /// Extensions for ChatOptions to support Anthropic-specific features
    /// </summary>
    public static class ChatOptionsExtensions
    {
        private const string ThinkingParametersKey = "Anthropic.ThinkingParameters";

        /// <summary>
        /// Sets thinking parameters for extended thinking support in compatible models like Claude 3.7 Sonnet
        /// </summary>
        /// <param name="options">The ChatOptions instance</param>
        /// <param name="budgetTokens">The budget tokens for thinking (typically up to max_tokens unless using interleaved thinking)</param>
        /// <returns>The ChatOptions instance for fluent chaining</returns>
        public static ChatOptions WithThinking(this ChatOptions options, int budgetTokens)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (budgetTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(budgetTokens), "Budget tokens must be greater than 0");

            (options.AdditionalProperties ??= new())[ThinkingParametersKey] = new ThinkingParameters
            {
                BudgetTokens = budgetTokens
            };

            return options;
        }

        /// <summary>
        /// Sets thinking parameters for extended thinking support in compatible models like Claude 3.7 Sonnet
        /// </summary>
        /// <param name="options">The ChatOptions instance</param>
        /// <param name="thinkingParameters">The thinking parameters to set</param>
        /// <returns>The ChatOptions instance for fluent chaining</returns>
        public static ChatOptions WithThinking(this ChatOptions options, ThinkingParameters thinkingParameters)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (thinkingParameters == null)
                throw new ArgumentNullException(nameof(thinkingParameters));

            (options.AdditionalProperties ??= new())[ThinkingParametersKey] = thinkingParameters;

            return options;
        }

        /// <summary>
        /// Sets interleaved thinking parameters for enhanced thinking support. This enables the interleaved-thinking-2025-05-14 beta header
        /// which allows thinking tokens to exceed max_tokens. Note: On 3rd-party platforms (Bedrock, Vertex AI), this only works 
        /// with Claude Opus 4.1, Opus 4, or Sonnet 4 models.
        /// </summary>
        /// <param name="options">The ChatOptions instance</param>
        /// <param name="budgetTokens">The budget tokens for thinking (can exceed max_tokens when using interleaved thinking)</param>
        /// <returns>The ChatOptions instance for fluent chaining</returns>
        public static ChatOptions WithInterleavedThinking(this ChatOptions options, int budgetTokens)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (budgetTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(budgetTokens), "Budget tokens must be greater than 0");

            (options.AdditionalProperties ??= new())[ThinkingParametersKey] = new ThinkingParameters
            {
                BudgetTokens = budgetTokens,
                UseInterleavedThinking = true
            };

            return options;
        }

        /// <summary>
        /// Sets interleaved thinking parameters for enhanced thinking support. This enables the interleaved-thinking-2025-05-14 beta header
        /// which allows thinking tokens to exceed max_tokens. Note: On 3rd-party platforms (Bedrock, Vertex AI), this only works 
        /// with Claude Opus 4.1, Opus 4, or Sonnet 4 models.
        /// </summary>
        /// <param name="options">The ChatOptions instance</param>
        /// <param name="thinkingParameters">The thinking parameters to set</param>
        /// <returns>The ChatOptions instance for fluent chaining</returns>
        public static ChatOptions WithInterleavedThinking(this ChatOptions options, ThinkingParameters thinkingParameters)
        {
            if (options == null)
                throw new ArgumentNullException(nameof(options));

            if (thinkingParameters == null)
                throw new ArgumentNullException(nameof(thinkingParameters));

            thinkingParameters.UseInterleavedThinking = true;
            (options.AdditionalProperties ??= new())[ThinkingParametersKey] = thinkingParameters;

            return options;
        }

        /// <summary>
        /// Gets the thinking parameters from ChatOptions
        /// </summary>
        /// <param name="options">The ChatOptions instance</param>
        /// <returns>The thinking parameters, or null if not set</returns>
        public static ThinkingParameters GetThinkingParameters(this ChatOptions options)
        {
            if (options?.AdditionalProperties?.TryGetValue(ThinkingParametersKey, out var value) == true)
            {
                return value as ThinkingParameters;
            }

            return null;
        }
    }
}