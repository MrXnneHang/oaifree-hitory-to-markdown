from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

data_path = Path("./data/messages-67415948-40a8-800b-b1ba-13a774e87e94.json")
output_path = Path("./output/template.md")
data = json.load(data_path.open())
conversation_len = len(data["messages"])

user_message = ""
response = ""
write_content_list = []
for i in range(conversation_len):
    if data["messages"][i]["author"]["role"] == "user":
        # user message
        user_message = data["messages"][i]["content"]["parts"][0]
        user_message_list = user_message.split("\n")
        ## 用**包裹并且去除空行以及空格
        user_message = "".join(
            "__" + x.strip() + "__<br>\r\n" for x in user_message_list if x
        )
        user_message = (
            user_message + "\r\n<details>\r\n<summary>\r\n回答\r\n</summary>\r\n\r\n"
        )
    elif data["messages"][i]["author"]["role"] == "assistant":
        # response
        response = data["messages"][i]["content"]["parts"][0]
        response = response + "\r\n\r\n</details>\r\n\r\n"
        write_content = user_message + response
        write_content_list.append(write_content)

output_path.write_text("\r\n".join(write_content_list))
