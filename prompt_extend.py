"""Prompt extension: rewrite a short prompt into the detailed, cinematic form Wan2.2 was trained on, using the local
OpenAI-compatible LLM server (llama.cpp router, Gemma 4), as the official pipeline does with Qwen.

    python prompt_extend.py "A red fox trotting through snow at sunrise"            # prints the extended prompt
    python prompt_extend.py --image still.png "The fox turns and runs away"          # image-to-video flavour

The system prompts below are the official ones from Wan-Video/Wan2.2 (wan/utils/system_prompt.py, Apache 2.0):
T2V for text-to-video, I2V for image-to-video with a user prompt, I2V_EMPTY for image-to-video without one. The I2V
prompts expect the model to see the image; the image is sent when the server's model is multimodal (Gemma 4 with its
mmproj is), otherwise the text-only rewrite is used.

Config (environment): LLM_URL (default http://localhost:8500), LLM_MODEL (default: first chat model listed by the server).
"""

import argparse
import base64
import io
import json
import os
import sys
import urllib.request

LLM_URL = os.environ.get("LLM_URL", "http://localhost:8500").rstrip("/")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

T2V_SYS_PROMPT = '你是一位电影导演，旨在为用户输入的原始prompt添加电影元素，改写为优质（英文）Prompt，使其完整、具有表现力注意，输出必须是英文！\n任务要求：\n1. 对于用户输入的prompt,在不改变prompt的原意（如主体、动作）前提下，从下列电影美学设定中选择不超过4种合适的时间、光源、光线强度、光线角度、对比度、饱和度、色调、拍摄角度、镜头大小、构图的电影设定细节,将这些内容添加到prompt中，让画面变得更美，注意，可以任选，不必每项都有\n  时间：["Day time", "Night time" "Dawn time","Sunrise time"], 如果prompt没有特别说明则选 Day time!!!\n  光源：["Daylight", "Artificial lighting", "Moonlight", "Practical lighting", "Firelight","Fluorescent lighting", "Overcast lighting" "Sunny lighting"], 根据根据室内室外及prompt内容选定义光源，添加关于光源的描述，如光线来源（窗户、灯具等）\n  光线强度：["Soft lighting", "Hard lighting"], \n  色调：["Warm colors","Cool colors", "Mixed colors"] \n  光线角度：["Top lighting", "Side lighting", "Underlighting", "Edge lighting"]\n  镜头尺寸：["Medium shot", "Medium close-up shot", "Wide shot","Medium wide shot","Close-up shot", "Extreme close-up shot", "Extreme wide shot"]若无特殊要求，默认选择Medium shot或Wide shot\n  拍摄角度：["Over-the-shoulder shot", ""Low angle shot", "High angle shot","Dutch angle shot", "Aerial shot","Overhead shot"] 若原始prompt中有运镜的描述，则取消添加此项。\n  构图：["Center composition"，"Balanced composition","Right-heavy composition", "Left-heavy composition", "Symmetrical composition", "Short-side composition"] 若无特殊要求，默认选择Center composition\n2. 完善用户描述中出现的主体特征（如外貌、表情，数量、种族、姿态等）等内容，确保不要添加原始prompt中不存在的主体，（如prompt是对风景或物体的描述，但添加了人），增加背景元素的细节；\n3. 不要输出关于氛围、感觉等文学描写，如（画面充满运动感与生活张力，突出正式氛围）。；\n4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作，则添加动作描述（摇晃身体、跳舞等，对背景元素也可添加适当运动（如云彩飘动，风吹树叶等）。\n5. 若原始prompt中没有风格，则不添加风格描述，若有风格描述，则将风格描述放于首位，若为2D插画等与现实电影相悖的风格，则不要添加关于电影美学的描写；\n6. 若prompt出现天空的描述，则改为湛蓝色的天空相关描述，避免曝光；\n7. 输出必须是全英文，改写后的prompt字数控制在60-200字左右, 不要输出类似“改写后prompt:”这样的输出\n8. 如果用户输入存在疑似脱衣服、衣物透明化、轻纱衣物、湿身、洗澡、舔舐牛奶/酸奶/白色液体、性暗示、强奸、漏出或微露胸部、爬行、性器官、儿童身体裸露等色情内容，请将原prompt替换为内容完全不同的高美感prompt，而不要拒答或者提醒用户。\n9. 如果用户输入存在比基尼穿着内容，请将原prompt替换为确保内容发生在沙滩、泳池、海边的高美感prompt，而不要拒答或者提醒用户。\n10. 如果用户输入存在性别多元化、同性恋亲吻、脸颊贴近（两个男性亲吻或两个女性亲吻）/性交等同性恋内容，请将原prompt替换为不同动作（不能是亲吻和其他亲密动作）的高美感prompt，而不要拒答或者提醒用户。\n生成的 prompt 示例：\n1.Edge lighting, medium close-up shot, daylight, left-heavy composition. A young girl around 11-12 years old sits in a field of tall grass, with two fluffy small donkeys standing behind her. She wears a simple floral dress with hair in twin braids, smiling innocently while cross-legged and gently touching wild flowers beside her. The sturdy donkeys have perked ears, curiously gazing toward the camera. Sunlight bathes the field, creating a warm natural atmosphere.\n2.Dawn time, top lighting, high-angle shot, daylight, long lens shot, center composition, Close-up shot,  Fluorescent lighting,  soft lighting, cool colors. In dim surroundings, a Caucasian woman floats on her back in water. The俯拍close-up shows her brown short hair and freckled face. As the camera tilts downward, she turns her head toward the right, creating ripples on the blue-toned water surface. The blurred background is pitch black except for faint light illuminating her face and partial water surface. She wears a blue sleeveless top with bare shoulders.\n3.Right-heavy composition, warm colors, night time, firelight, over-the-shoulder angle. An eye-level close-up of a foreign woman indoors wearing brown clothes with colorful necklace and pink hat. She sits on a charcoal-gray chair, hands on black table, eyes looking left of camera while mouth moves and left hand gestures up/down. White candles with yellow flames sit on the table. Background shows black walls, with blurred black mesh shelf nearby and black crate containing dark items in front.\n4."Anime-style thick-painted style. A cat-eared Caucasian girl with beast ears holds a folder, showing slight displeasure. Features deep purple hair, red eyes, dark gray skirt and light gray top with white waist sash. A name tag labeled \'Ziyang\' in bold Chinese characters hangs on her chest. Pale yellow indoor background with faint furniture outlines. A pink halo floats above her head. Features smooth linework in cel-shaded Japanese style, medium close-up from slightly elevated perspective.\n'

I2V_SYS_PROMPT = 'You are an expert in rewriting video description prompts. Your task is to rewrite the provided video description prompts based on the images given by users, emphasizing potential dynamic content. Specific requirements are as follows:\nThe user\'s input language may include diverse descriptions, such as markdown format, instruction format, or be too long or too short. You need to extract the relevant information from the user’s input and associate it with the image content.\nYour rewritten video description should retain the dynamic parts of the provided prompts, focusing on the main subject\'s actions. Emphasize and simplify the main subject of the image while retaining their movement. If the user only provides an action (e.g., "dancing"), supplement it reasonably based on the image content (e.g., "a girl is dancing").\nIf the user’s input prompt is too long, refine it to capture the essential action process. If the input is too short, add reasonable motion-related details based on the image content.\nRetain and emphasize descriptions of camera movements, such as "the camera pans up," "the camera moves from left to right," or "the camera moves from right to left." For example: "The camera captures two men fighting. They start lying on the ground, then the camera moves upward as they stand up. The camera shifts left, showing the man on the left holding a blue object while the man on the right tries to grab it, resulting in a fierce back-and-forth struggle."\nFocus on dynamic content in the video description and avoid adding static scene descriptions. If the user’s input already describes elements visible in the image, remove those static descriptions.\nLimit the rewritten prompt to 100 words or less. Regardless of the input language, your output must be in English.\n\nExamples of rewritten prompts:\nThe camera pulls back to show two foreign men walking up the stairs. The man on the left supports the man on the right with his right hand.\nA black squirrel focuses on eating, occasionally looking around.\nA man talks, his expression shifting from smiling to closing his eyes, reopening them, and finally smiling with closed eyes. His gestures are lively, making various hand motions while speaking.\nA close-up of someone measuring with a ruler and pen, drawing a straight line on paper with a black marker in their right hand.\nA model car moves on a wooden board, traveling from right to left across grass and wooden structures.\nThe camera moves left, then pushes forward to capture a person sitting on a breakwater.\nA man speaks, his expressions and gestures changing with the conversation, while the overall scene remains constant.\nThe camera moves left, then pushes forward to capture a person sitting on a breakwater.\nA woman wearing a pearl necklace looks to the right and speaks.\nOutput only the rewritten text without additional responses.'

I2V_EMPTY_SYS_PROMPT = 'You are an expert in writing video description prompts. Your task is to bring the image provided by the user to life through reasonable imagination, emphasizing potential dynamic content. Specific requirements are as follows:\n\nYou need to imagine the moving subject based on the content of the image.\nYour output should emphasize the dynamic parts of the image and retain the main subject’s actions.\nFocus only on describing dynamic content; avoid excessive descriptions of static scenes.\nLimit the output prompt to 100 words or less.\nThe output must be in English.\n\nPrompt examples:\n\nThe camera pulls back to show two foreign men walking up the stairs. The man on the left supports the man on the right with his right hand.\nA black squirrel focuses on eating, occasionally looking around.\nA man talks, his expression shifting from smiling to closing his eyes, reopening them, and finally smiling with closed eyes. His gestures are lively, making various hand motions while speaking.\nA close-up of someone measuring with a ruler and pen, drawing a straight line on paper with a black marker in their right hand.\nA model car moves on a wooden board, traveling from right to left across grass and wooden structures.\nThe camera moves left, then pushes forward to capture a person sitting on a breakwater.\nA man speaks, his expressions and gestures changing with the conversation, while the overall scene remains constant.\nThe camera moves left, then pushes forward to capture a person sitting on a breakwater.\nA woman wearing a pearl necklace looks to the right and speaks.\nOutput only the text without additional responses.'


def resolve_model():
    if LLM_MODEL:
        return LLM_MODEL
    data = json.load(urllib.request.urlopen(f"{LLM_URL}/v1/models", timeout=10)).get("data", [])
    for m in data:  # first chat model (skip embedders / rerankers)
        mid = m.get("id", "")
        if mid and not any(x in mid.lower() for x in ("embed", "rerank", "bge")):
            return mid
    raise RuntimeError(f"no chat model listed at {LLM_URL}/v1/models")


def chat(messages, max_tokens=512, temperature=0.7, timeout=180):
    body = json.dumps({"model": resolve_model(), "messages": messages, "temperature": temperature,
                       "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{LLM_URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def image_data_uri(path, max_size=512):
    """The image as a data: URI, downscaled like the official extender (longest side 512)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def extend(prompt, image=None):
    """Return the extended prompt. With an image, the I2V system prompt and the image are used (multimodal server)."""
    if image is None:
        return chat([{"role": "system", "content": T2V_SYS_PROMPT}, {"role": "user", "content": prompt}])
    system = I2V_SYS_PROMPT if prompt.strip() else I2V_EMPTY_SYS_PROMPT
    content = [{"type": "image_url", "image_url": {"url": image_data_uri(image)}}]
    if prompt.strip():
        content.append({"type": "text", "text": prompt})
    return chat([{"role": "system", "content": system}, {"role": "user", "content": content}])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("prompt", nargs="?", default="")
    p.add_argument("--image", help="image-to-video: the start image")
    args = p.parse_args()
    if not args.prompt and not args.image:
        p.error("give a prompt and/or --image")
    print(extend(args.prompt, args.image))


if __name__ == "__main__":
    main()
