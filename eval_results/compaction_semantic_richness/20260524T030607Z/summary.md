# Compaction Semantic Richness Eval

Output directory: `eval_results/compaction_semantic_richness/20260524T030607Z`
Cases scored: 2 / 3
Errors: 1
Average semantic richness score: 1.0

| Case | Score | Atoms | Sections | Revisions |
| --- | ---: | ---: | ---: | ---: |
| cli_telemetry_resource_exhaustion | 1.0000 | 9/9 | 12/12 | 1 |
| base_chat_model_setup_and_compaction | 1.0000 | 9/9 | 12/12 | 0 |

## Errors

- `setup_script_dependency_bootstrap`: ChatGoogleGenerativeAIError: Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.5-flash\nPlease retry in 31.387850083s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-3.5-flash', 'location': 'global'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '31s'}]}}
