;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "chapter2"
 (lambda ()
   (LaTeX-add-labels
    "chap:ch1"
    "FigCBSD"
    "TabelSolutii"
    "LabelMyEquation"))
 :latex)

