from __future__ import unicode_literals

import re


def removeCodeAndHTML(txt):
    txt = removeHTML(txt)

    txt = removeCode(txt)
    # dlog(txt)
    return txt


def removeCode(text):
    if "<code>" or "<blockquote>" in text:

        pattern = re.compile("<code>(.+?)</code>")

        for m in re.finditer(pattern, text):

            found = m.group(1)

            if len(found.split()) == 1:
                if (
                    "?" in found
                    or "\\" in found
                    or "{" in found
                    or "}" in found
                    or "*" in found
                    or "]" in found
                    or "[" in found
                    or "+" in found
                ):
                    text = re.sub("<code>(.+?)</code>", "", text)
                    # log(found)
                else:
                    txt = m.group(0)
                    txt = re.sub(txt, found, text)

            elif len(found.split()) > 1:
                text = re.sub("<code>(.+?)</code>", "", text)

    return text


def removeHTML(text):

    text = text.replace("\n", "")
    # text = re.sub('\n', '');

    text = re.sub("<a (.+?)>", "", text)
    text = re.sub("</a>", "", text)
    text = re.sub("<img (.+?)>", "", text)

    text = text.replace("&#xA;", " ")

    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("<p>", "")
    text = text.replace("</p>", "")
    text = text.replace("<br>", "")
    text = text.replace("</br>", "")
    text = text.replace("<br/>", "")
    text = text.replace("<strong>", "")
    text = text.replace("</strong>", "")
    text = text.replace("<blockquote>", "")
    text = text.replace("</blockquote>", "")
    text = text.replace("<pre>", "")
    text = text.replace("</pre>", "")
    text = text.replace("<em>", "")
    text = text.replace("</em>", "")
    text = text.replace("<ul>", "")
    text = text.replace("</ul>", "")
    text = text.replace("<li>", "")
    text = text.replace("</li>", "")
    text = text.replace("<h1>", "")
    text = text.replace("</h1>", "")
    text = text.replace("<h2>", "")
    text = text.replace("</h2>", "")
    text = text.replace("<h3>", "")
    text = text.replace("</h3>", "")
    text = text.replace("<hr>", "")
    text = text.replace("<ol>", "")
    text = text.replace("(", "")
    text = text.replace(")", "")

    return text
