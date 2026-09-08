---
title: "Markdown Format Test"
date: 2025-03-23
draft: false
tags: ["markdown", "test"]
categories: ["Misc"]
language: en
# 主题演示页：保留可访问，但不出现在列表 / RSS / sitemap 中
build:
  list: never
---

```callout {.neutral title="About this page"}
This is a **theme demo page** for [hugo-theme-void](https://github.com/Daucloud/hugo-theme-void). It exercises every Markdown element the theme styles, so it is kept online for reference but hidden from the post list, RSS feed and sitemap.
```

# Markdown Format Test

This article demonstrates how various Markdown format elements render on this blog.

## 1. Basic Text Formatting

**Bold text** and _italic text_

**_Bold and italic text_**

~~Strikethrough text~~

<u>Underlined text</u>

## 2. Lists

### Unordered Lists

- Item 1
- Item 2
  - Sub-item 2.1
  - Sub-item 2.2
- Item 3

### Ordered Lists

1. First step
2. Second step
   1. Sub-step 2.1
   2. Sub-step 2.2
3. Third step

### Task Lists

- [x] Completed task
- [ ] Incomplete task
- [ ] Another incomplete task

## 3. Links and Images

[Hugo Official Website](https://gohugo.io/)

![Hugo Logo](https://d33wubrfki0l68.cloudfront.net/c38c7334cc3f23585738e40334284fddcaf03d5e/2e17c/images/hugo-logo-wide.svg)

## 4. Blockquotes

> This is a quoted text.
>
> This is the second paragraph of the quote.

## 5. Code

Inline code `print("Hello, World!")`

Code blocks:

```python
def greet(name):
    """
    A simple greeting function
    """
    print(f"Hello, {name}!")

# Call the function
greet("World")
```

```javascript
// JavaScript code example
function calculateSum(a, b) {
  return a + b;
}

console.log(calculateSum(5, 10)); // Output: 15
```

## 6. Tables

| Name | Age | Profession      |
| ---- | --- | --------------- |
| John | 25  | Engineer        |
| Lisa | 30  | Designer        |
| Mike | 28  | Product Manager |

## 7. Horizontal Rules

---

## 8. Math Equations

Inline equation: $E = mc^2$

Block equation:

$$
\frac{d}{dx}(x^n) = nx^{n-1}
$$

## 9. Footnotes

This is a text with a footnote[^1].

[^1]: This is the footnote content.

## 10. Definition Lists

Term 1
: Definition 1

Term 2
: Definition 2a
: Definition 2b
