# Open-source PhD Thesis

[![Website](https://img.shields.io/badge/Thesis-Visit-blue)](https://phd.dramsch.net/phd)
[![PDF Download](https://img.shields.io/badge/PDF-Download-important)](/files/Dramsch-thesis.pdf)
[![Epub Download](https://img.shields.io/badge/Epub-Download-green)](/files/dramsch-phd-thesis.epub)
![YouTube Video Views](https://img.shields.io/youtube/views/aNXyx215brU?style=flat-square&label=Defense%20views&link=https%3A%2F%2Fyoutu.be%2FaNXyx215brU)
[![GitHub](https://img.shields.io/github/license/jesperdramsch/phd-thesis)](https://github.com/jesperdramsch/phd-thesis/blob/thesis/LICENSE)

## Table of Contents

- [Open-source PhD Thesis](#open-source-phd-thesis)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Recreate this Project](#recreate-this-project)
  - [Usage](#usage)
  - [License](#license)

## Introduction

A webview of the thesis “Machine Learning in Geoscience” by [Jesper Dramsch](https://dramsch.net).

This project uses [pixi](https://pixi.sh) to manage the project and [nikola](https://getnikola.com) to generate the website. The website itself is static and can be hosted on any web server, including Github Pages. The original thesis was written in LaTeX and converted to restructured text using [pandoc](https://pandoc.org). The website is generated using the [nikola](https://getnikola.com) static site generator.

You can read my [online thesis](https://phd.dramsch.net/phd) and [download the pdf](/files/Dramsch-thesis.pdf).

## Recreate this Project

1. Write a PhD thesis.
2. Install [pixi](https://pixi.sh) to manage dependencies.
3. Use [pandoc](https://pandoc.org) to convert the thesis chapters to restructured text or markdown.
4. Fix all the errors from pandoc (e.g. missing references, wrong formatting, etc.), and add the necessary metadata. (This is due to a thesis being fairly complex and to no fault of pandoc, which is amazing.)
5. Use [nikola](https://getnikola.com) to generate the website.
6. _Optionally_, use [pandoc](https://pandoc.org) to convert the thesis to an epub.
7. Create an extra page that uses [shields.io](https://shields.io) and other toys to display all [code contributions](/pages/code.rst) in the thesis.
8. _Optionally_, defend your PhD and record the presentation, [upload it to YouTube](https://youtu.be/aNXyx215brU) and embed it in the website.
9. Tell people about it!

## Usage

To use the project make sure you have [pixi](https://pixi.sh) installed, then follow these steps:

```bash
git clone https://github.com/jesperdramsch/phd-thesis.git
cd phd-thesis
pixi serve -b
```

## License

Part of this project is licensed under a modified MIT License - see the [LICENSE](/LICENSE) file for details. This explicitly excludes the content of the thesis, which is complicated due to the nature of academic publishing. The code and website are open-source, and you are welcome to use them for your own thesis. If you do, please let me know, I would love to see what you come up with!
