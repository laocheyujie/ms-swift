from swift.llm import get_model_tokenizer, get_template

_, tokenizer = get_model_tokenizer('/models/ZhipuAI/GLM-4.5-Air', load_model=False)
template = get_template(
    tokenizer.model_meta.template, 
    tokenizer, 
    # agent_template='hermes'
)
data = {
    "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", 
    "messages": [
        {"role": "user", "content": "北京和上海今天的天气情况"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, 
        {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"},
        # {"role": "tool_response", "content": "北京挺好，上海挺好"},
        # {"role": "assistant", "content": "北京和上海今天的天气挺好"}
    ]
}
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')