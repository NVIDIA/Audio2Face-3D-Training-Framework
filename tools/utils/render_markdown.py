# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import argparse
import html
import re
import markdown2


def convert_markdown_to_html(md_filepath: str, html_filepath: str, title: str = "") -> None:
    with open(md_filepath, "r", encoding="utf-8") as md_file:
        md_content = md_file.read()

    link_patterns = [(re.compile(r"\b(http://\S+|https://\S+)"), r"\1")]
    html_content = markdown2.markdown(md_content, extras={"fenced-code-blocks": {}, "link-patterns": link_patterns})
    escaped_title = html.escape(title)

    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{escaped_title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.3.0/github-markdown-dark.css">
    <style>
        body {{
            background-color: #0D1117;
            color: #C9D1D9;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
        }}
        h1 {{
            color: #76b900;
        }}
        h2 {{
            color: #76b900;
        }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>
"""

    with open(html_filepath, "w", encoding="utf-8") as html_file:
        html_file.write(html_template)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Markdown to HTML with styling")
    parser.add_argument("md_filepath", help="Path to the input Markdown file")
    parser.add_argument("-o", "--output", dest="html_filepath", help="Path to the output HTML file")
    parser.add_argument("-t", "--title", help="Title for the HTML document")
    args = parser.parse_args()

    if args.html_filepath is None:
        args.html_filepath = os.path.splitext(args.md_filepath)[0] + ".html"
    if args.title is None:
        args.title = os.path.splitext(os.path.basename(args.md_filepath))[0]

    convert_markdown_to_html(args.md_filepath, args.html_filepath, args.title)
    print(f"Converted {args.md_filepath} to {args.html_filepath}")


if __name__ == "__main__":
    main()
