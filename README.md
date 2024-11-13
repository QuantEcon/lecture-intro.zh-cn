# lecture-intro.zh-cn

An Undergraduate Lecture Series for the Foundations of Computational Economics (Chinese Language)

## Installing the quantecon environment

When working on the lectures it is useful to install the quantecon environment for executing and building 
the lectures locally. 

This can be done using conda by:

```bash
conda env create -f environment.yml
```

or within China it is best to use the `environment-cn.yml`

```bash
conda env create -f environment-cn.yml
```

---

## Translation Project Plan

### Description

This repo is a testing ground for the development of a high-quality Chinese language version of the
introductory lecture site. 

This project will initially undertake a manual editing process (with the assistance of AI to generate the new source material)
with the eventual goal of automating as much of the conversion process as possible. This will include iterating and tracking 
changes to see if we can (a) improve the initial AI conversion, and (b) think about the future editing experience for improvements
to future workflows (ie. how to keep the english and chinese versions in sync ENGLISH -> CHINESE; ENGLISH<->CHINESE)

### Phase 1 (Short Term)

**Aim:** To deliver a high quality Chinese language version of the [intro lecture series](https://intro.quantecon.org/intro.html)

| Role | Lead | 
|------|----------|
| ChatGPT / AI | Humphrey |
| Editorial and Workflow  | Longye and Sylvia |
| GitHub Integration | Matt |

**Schedule:**

| Target Date | Milestone |
|-------------|-----------|
| 15th September 2024 | Jupyter Book powered version building with HTML |
| 30th September 2024 | Version 1 to share more broadly with teams |
| Oct - Nov 2024 | Technical refinements in editorial workflow process (identify future software development needs) |

**Organisation of Repo:**

- any conversion programs should be contained in `converter/` and documentation in `converter/README.md`
- repo should mirror english language as much as possible
  - lectures are in `lectures` etc.


### Phase 2 (Medium Term)

**Aim:** To improve workflows to increase automation to allow for translations to be more automated, and improve the ability to maintain different language versions and keep them in sync.

Improvements to:

1. Uni-directional syncing (ENGLISH -> CHINESE)
2. Bi-directional syncing (ENGLISH -> CHINESE and CHINESE -> ENGLISH)


### Phase 3 (Longer Term)

**Aim:** Generalise what we have learnt on maintaining different languages and work with broader open-source community to release tools that enable other sites to maintain multiple language sites. 