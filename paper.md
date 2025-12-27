---
title: "Jupyter Course Assistant: A Notebook-Native Agentic Teaching Assistant for Technical Courses"
authors:
  - name: Faiyaz Azam
    affiliation: Department of Mechanical Engineering, Carnegie Mellon University
    email: fazam@andrew.cmu.edu
date: 2025-09-XX
license: MIT
---

## Summary

Jupyter notebooks are widely used in engineering and scientific education as a medium for coding, mathematical derivations, and written explanations. However, students often struggle to connect course readings, theoretical derivations, and their own notebook-based work in a coherent way. This paper introduces **Jupyter Course Assistant**, an open-source, Jupyter-native teaching assistant that enables students to interact directly with course materials and receive structured feedback within their notebooks. The system combines retrieval-augmented generation (RAG) over instructor-provided PDFs with a notebook-aware analysis tool, allowing students to ask conceptual questions, receive LaTeX-rendered explanations, and obtain suggestions for improving their notebooks. The tool is designed to be lightweight, reproducible, and pedagogically aligned with notebook-based technical courses.

## Statement of Need

Notebook-centric workflows are now standard in many undergraduate and graduate technical courses, particularly in optimization, machine learning, robotics, and data science. While large language models have the potential to support learning in these settings, existing tools often operate outside the notebook environment, require repeated document processing, or blur the distinction between conceptual assistance and solution generation. There is a need for an educationally grounded tool that (i) integrates seamlessly into Jupyter notebooks, (ii) respects instructor-curated course materials, and (iii) supports student understanding without replacing independent problem-solving. Jupyter Course Assistant addresses this need by providing a notebook-native interface for interacting with course readings and reflecting on notebook content itself with citations that lead back to course content provided by instructors, enhancing learning.

## Design and Implementation

The Jupyter Course Assistant is implemented as a custom IPython magic extension, allowing students to interact with the agent using simple commands (e.g., `%%research_agent`, `%ask notebook`) directly inside a Jupyter notebook. The system distinguishes between two roles:

**Instructor workflow.** Instructors build a persistent course memory by placing PDFs (e.g., lecture notes, papers) into a designated directory and running a helper script that constructs a vector index using text embeddings. This step is performed once and the resulting index is committed to the repository.

**Student workflow.** Students load the extension and initialize the agent without any preprocessing or embedding steps. Queries are answered using retrieval over the prebuilt course index combined with a language model for reasoning. Mathematical expressions are rendered using LaTeX to maintain clarity and consistency with technical coursework.

In addition to course question answering, the assistant includes a *notebook inspector* tool that indexes the content of the active notebook. This enables students to ask questions about their own work, such as requesting summaries, identifying unclear sections, or receiving high-level suggestions for improvement. The implementation relies on open-source libraries, including LlamaIndex for retrieval infrastructure and Jupyter/IPython for notebook integration.

## Educational Use Cases

The tool has been approved for use in a graduate-level optimization course, *24-786 Advanced Optimization for Engineering*, at Carnegie Mellon University. In this context, students use the assistant to explore conceptual questions related to Alternating Direction Method of Multipliers (ADMM) derivations, such as understanding update steps and constraints, while working through notebook-based assignments. The assistant was used by peers and approved by the course instructor as a supplementary learning aid. More broadly, the tool is applicable to mixed undergraduate and graduate technical courses that rely on Jupyter notebooks for coding, derivations, and written explanations.

## Related Work

Jupyter notebooks have been widely adopted as an educational medium for computational subjects. Prior work has explored the use of notebooks for reproducible research and interactive learning. Retrieval-augmented generation has recently been proposed as a way to ground language model outputs in source documents, but most existing systems are not designed for notebook-native educational use. Jupyter Course Assistant builds on these ideas by tightly integrating retrieval, reasoning, and notebook awareness in a form that is accessible to students and instructors.

## Availability

The Jupyter Course Assistant is released as open-source software under the MIT License. The source code, documentation, and example notebooks are publicly available at:

https://github.com/FaiyazAzam/jupyter-course-assistant

The repository includes instructions for instructors to build course memory and for students to use the assistant within their notebooks.

