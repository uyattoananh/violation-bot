const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

// ---------- style helpers ----------
const BRAND = "1F4E79";
const INK = "1A1A1A";
const MUTED = "595959";
const RULE = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const CELL_BORDERS = { top: RULE, bottom: RULE, left: RULE, right: RULE };
const CELL_MARGINS = { top: 100, bottom: 100, left: 140, right: 140 };

const p = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 120 },
    ...opts.paragraph,
    children: [new TextRun({ text, size: 22, color: INK, ...opts.run })],
  });

const pRich = (runs, opts = {}) =>
  new Paragraph({
    spacing: { after: 120 },
    ...opts.paragraph,
    children: runs.map((r) =>
      typeof r === "string"
        ? new TextRun({ text: r, size: 22, color: INK })
        : new TextRun({ size: 22, color: INK, ...r })
    ),
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, bold: true, size: 32, color: BRAND })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    children: [new TextRun({ text, bold: true, size: 26, color: BRAND })],
  });

const h3 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, bold: true, size: 23, color: INK })],
  });

const bullet = (text, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: [new TextRun({ text, size: 22, color: INK })],
  });

const bulletRich = (runs, level = 0) =>
  new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { after: 80 },
    children: runs.map((r) =>
      typeof r === "string"
        ? new TextRun({ text: r, size: 22, color: INK })
        : new TextRun({ size: 22, color: INK, ...r })
    ),
  });

// ---------- table helpers ----------
const CONTENT_WIDTH = 9360; // US Letter, 1" margins

const makeTable = (columnWidths, rows, opts = {}) => {
  if (columnWidths.reduce((a, b) => a + b, 0) !== CONTENT_WIDTH) {
    throw new Error("columnWidths must sum to " + CONTENT_WIDTH);
  }
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths,
    rows: rows.map((row, i) => {
      const isHeader = opts.header && i === 0;
      return new TableRow({
        children: row.map((cell, j) => {
          const content = Array.isArray(cell) ? cell : [cell];
          return new TableCell({
            borders: CELL_BORDERS,
            width: { size: columnWidths[j], type: WidthType.DXA },
            margins: CELL_MARGINS,
            shading: isHeader
              ? { fill: "D9E3F0", type: ShadingType.CLEAR }
              : undefined,
            children: content.map((c) =>
              typeof c === "string"
                ? new Paragraph({
                    children: [
                      new TextRun({
                        text: c,
                        size: 20,
                        bold: isHeader,
                        color: INK,
                      }),
                    ],
                  })
                : c
            ),
          });
        }),
      });
    }),
  });
};

// ---------- content ----------
const children = [];

// Title block
children.push(
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1800, after: 240 },
    children: [
      new TextRun({
        text: "AI Construction Violation Classifier",
        bold: true,
        size: 44,
        color: BRAND,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({
        text: "Pipeline Options, Cost Analysis & Recommendation",
        size: 28,
        color: MUTED,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({
        text: "Vietnamese Labor Safety Law · Project Management Company Deployment",
        italics: true,
        size: 22,
        color: MUTED,
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({ text: "Prepared: April 2026", size: 20, color: MUTED }),
    ],
  }),
  new Paragraph({ children: [new PageBreak()] })
);

// ---------- Executive Summary ----------
children.push(h1("1. Executive Summary"));
children.push(
  p(
    "This document evaluates five AI pipeline architectures for a web application that classifies construction safety violations from site photographs under Vietnamese labor safety law. The deployment context is a project management (PM) company overseeing ten concurrent large-scale projects (malls, hotels, hospitals, apartment buildings) with batch photo uploads performed after each site visit."
  )
);
children.push(
  p(
    "The expected workload of approximately 8,000 photos per month falls well within the range where cloud API inference dominates dedicated GPU infrastructure on total cost of ownership. The recommended starting architecture is a zero-shot Vision Language Model (VLM) pipeline grounded against the QCVN 18:2021/BXD violation taxonomy, with inspector feedback captured as a labeled dataset for a later graduation to a hybrid detector plus VLM pipeline once 2,000+ confirmed labels have been collected."
  )
);
children.push(h3("Bottom line"));
children.push(
  bulletRich([
    { text: "Recommended monthly cost: ", bold: true },
    "USD 170-210 all-in (AI inference + hosting + storage + database)",
  ])
);
children.push(
  bulletRich([
    { text: "Recommended architecture: ", bold: true },
    "Option 1 (Zero-shot VLM) for MVP; migrate to Option 3 (Hybrid) at month 9-12",
  ])
);
children.push(
  bulletRich([
    { text: "Dedicated GPU break-even: ", bold: true },
    "not reached until ~14,000 photos/month sustained — do not buy hardware",
  ])
);
children.push(
  bulletRich([
    { text: "Time to MVP: ", bold: true },
    "approximately 3 weeks for a working pilot on one project",
  ])
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Usage Assumptions ----------
children.push(h1("2. Usage Assumptions & Photo Volume"));
children.push(
  p(
    "Volume is the single most important driver of the cost comparison. Large-scale Vietnamese construction projects (malls, hotels, hospitals, apartment complexes) typically last 18-36 months, so all ten projects are assumed to run concurrently at steady state. Site safety inspections are conducted 2-5 times per week depending on project phase, with 30-80 photographs captured per walk-through."
  )
);
children.push(h3("Volume scenarios"));
children.push(
  makeTable(
    [2000, 1600, 1600, 1600, 2560],
    [
      [
        "Scenario",
        "Inspections/week",
        "Photos/inspection",
        "Projects concurrent",
        "Photos/month (≈)",
      ],
      ["Low (quiet phase)", "2", "40", "10", "3,400"],
      ["Baseline (recommended)", "3", "60", "10", "7,700"],
      ["High (peak construction)", "5", "80", "10", "17,200"],
    ],
    { header: true }
  )
);
children.push(
  p(
    "All subsequent cost figures use the baseline scenario of approximately 8,000 photos per month unless otherwise noted. Batch upload after site visits (rather than live streaming) means the system can use queued asynchronous inference rather than low-latency real-time processing, which further reduces infrastructure complexity and cost.",
    { paragraph: { spacing: { before: 120 } } }
  )
);

// ---------- Vietnamese Regulatory Scope ----------
children.push(h1("3. Vietnamese Regulatory Scope"));
children.push(
  p(
    "The classifier operates against the Vietnamese occupational safety and construction safety regulatory corpus. The following primary instruments must be encoded in the violation taxonomy and (for RAG-based variants) embedded as retrieval context."
  )
);
children.push(h3("Primary legal instruments"));
children.push(
  bulletRich([
    { text: "Law No. 84/2015/QH13", bold: true },
    " — Law on Occupational Safety and Hygiene (Luật An toàn, Vệ sinh Lao động)",
  ])
);
children.push(
  bulletRich([
    { text: "QCVN 18:2021/BXD", bold: true },
    " — National Technical Regulation on Safety in Construction (issued by the Ministry of Construction; the most operationally relevant standard for site-level violations)",
  ])
);
children.push(
  bulletRich([
    { text: "Decree 39/2016/ND-CP", bold: true },
    " — detailed provisions implementing the OSH Law",
  ])
);
children.push(
  bulletRich([
    { text: "Decree 44/2016/ND-CP", bold: true },
    " — occupational safety training, inspection, and technical requirements",
  ])
);
children.push(
  bulletRich([
    { text: "Circular 04/2017/TT-BXD", bold: true },
    " — Ministry of Construction detailed technical safety requirements",
  ])
);
children.push(
  bulletRich([
    { text: "TCVN 5308:1991 (where still referenced)", bold: true },
    " — legacy construction safety rules still cited in some contract specifications",
  ])
);

children.push(h3("Visual violation taxonomy (proposed, 12 top-level categories)"));
children.push(
  p(
    "Each category below is detectable from site photography and maps to one or more clauses in QCVN 18:2021/BXD or the OSH Law. This flat taxonomy is suitable for zero-shot VLM prompting; a deeper sub-clause taxonomy is needed only if the product outputs formal citations for enforcement."
  )
);
children.push(
  makeTable(
    [3500, 2500, 3360],
    [
      ["Category (EN)", "Category (VN)", "Primary authority"],
      ["PPE — hard hat", "Mũ bảo hộ", "QCVN 18:2021/BXD §2.1"],
      ["PPE — safety harness / fall arrest", "Dây an toàn", "QCVN 18:2021/BXD §2.17"],
      ["PPE — high-visibility clothing, boots, gloves", "Trang phục bảo hộ", "QCVN 18:2021/BXD §2.1"],
      ["Fall protection at height (>2 m)", "Làm việc trên cao", "QCVN 18:2021/BXD §2.17"],
      ["Scaffolding integrity", "Giàn giáo", "QCVN 18:2021/BXD §2.16"],
      ["Electrical hazards", "An toàn điện", "QCVN 18:2021/BXD §2.9"],
      ["Lifting equipment / cranes", "Thiết bị nâng", "QCVN 18:2021/BXD §2.12"],
      ["Excavation / trenching", "Đào đất, hố móng", "QCVN 18:2021/BXD §2.5"],
      ["Housekeeping / walkways", "Vệ sinh công trường", "QCVN 18:2021/BXD §2.2"],
      ["Fire safety / hot work", "Phòng chống cháy, hàn cắt", "QCVN 18:2021/BXD §2.11"],
      ["Material storage & stacking", "Lưu trữ vật tư", "QCVN 18:2021/BXD §2.3"],
      ["Safety signage (Vietnamese-language)", "Biển báo an toàn", "Decree 39/2016/ND-CP"],
    ],
    { header: true }
  )
);
children.push(
  p(
    "Important caveat: the mapping above is a working taxonomy, not a legal opinion. The final category-to-clause mapping should be reviewed by a Vietnamese-licensed occupational safety consultant before the product is used to produce any citation that could be relied on in a formal inspection or enforcement context.",
    { run: { italics: true, color: MUTED } }
  )
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Pipeline Options ----------
children.push(h1("4. Pipeline Options"));
children.push(
  p(
    "Five architectures are evaluated. Each is described in terms of its data flow, strengths, limitations, and fit for the ten-project PM-company context."
  )
);

// Option 1
children.push(h2("Option 1 — Zero-shot VLM (Vision Language Model)"));
children.push(
  p(
    "User uploads a photo in batch. The image is sent to a hosted VLM (Claude Sonnet/Haiku, Gemini Flash, or GPT-4o) along with a structured prompt containing the 12-category Vietnamese violation taxonomy. The model returns a JSON payload: label, confidence, reasoning, and bounding-box or pixel-region hints. Results are written to Postgres and surfaced in the web UI for inspector review."
  )
);
children.push(h3("Strengths"));
children.push(bullet("No training data required; operational on day one."));
children.push(bullet("New violation categories can be added by editing the prompt."));
children.push(bullet("Explainable output — the model provides reasoning a human can audit."));
children.push(bullet("Lowest infrastructure footprint; scales elastically with no capacity planning."));
children.push(h3("Limitations"));
children.push(bullet("Per-image API cost dominates at very high volumes."));
children.push(bullet("Accuracy is entirely prompt-engineering dependent; no model specialization for Vietnamese sites."));
children.push(bullet("Vendor lock-in on the model provider; pricing can change."));
children.push(bullet("2-5 second latency per image — acceptable for batch upload, not for live use."));

// Option 2
children.push(h2("Option 2 — Fine-tuned Vision Classifier"));
children.push(
  p(
    "Collect a labeled dataset of Vietnamese construction-site photographs (2,000-10,000 images across the 12 categories). Fine-tune a vision model — YOLOv8 or RT-DETR for detection with bounding boxes, or CLIP/ViT for classification. Deploy as an ONNX model served from a small GPU instance or, for YOLO-sized models, a CPU instance."
  )
);
children.push(h3("Strengths"));
children.push(bullet("Very cheap at scale once deployed — inference cost approaches zero per image."));
children.push(bullet("Low latency (<200 ms/image); runs on-device or on modest cloud GPUs."));
children.push(bullet("Produces bounding boxes for multi-violation scenes."));
children.push(bullet("No external API dependency — full data sovereignty."));
children.push(h3("Limitations"));
children.push(bullet("Requires 2k-10k labeled images, costing USD 5,000-50,000 in labeling labor or equivalent internal effort."));
children.push(bullet("Retraining cycle needed whenever the violation taxonomy or dominant site conditions change."));
children.push(bullet("Premature for this project — the correct label set is not yet known."));
children.push(bullet("MLOps overhead (model registry, versioning, monitoring) is non-trivial."));

// Option 3
children.push(h2("Option 3 — Hybrid: Detector + VLM Reasoner"));
children.push(
  p(
    "A fine-tuned object detector (YOLOv8 class) identifies atomic site elements — worker, hard hat, harness, ladder, guardrail, ladder rung, extension cord, etc. A deterministic rules engine (or a small LLM) evaluates combinations of detections against the Vietnamese violation taxonomy. Ambiguous scenes that the rules engine flags as low-confidence fall back to a full VLM call (Claude Sonnet) for a reasoned verdict."
  )
);
children.push(h3("Strengths"));
children.push(bullet("Strongest accuracy/cost balance once the detector is trained."));
children.push(bullet("Auditable: the rules engine produces deterministic, reviewable citations."));
children.push(bullet("Handles novel contexts via the VLM fallback path."));
children.push(bullet("VLM spend drops by 10-20x relative to Option 1 because only ambiguous images escalate."));
children.push(h3("Limitations"));
children.push(bullet("Two training pipelines (detector + rules taxonomy authoring)."));
children.push(bullet("Rules engine itself requires occupational-safety domain knowledge to author well."));
children.push(bullet("Higher upfront engineering cost than Option 1."));

// Option 4
children.push(h2("Option 4 — RAG-Grounded VLM (Citation-First)"));
children.push(
  p(
    "The Vietnamese safety regulatory corpus (QCVN 18:2021/BXD, Law 84/2015/QH13, associated decrees and circulars) is parsed into clause-level chunks and embedded into a vector database (pgvector). For each uploaded photo, the system retrieves the top-k most relevant clauses based on image features, then asks the VLM to classify the image against the retrieved clauses specifically — producing not just a label but a verbatim regulatory citation."
  )
);
children.push(h3("Strengths"));
children.push(bullet("Every label is tied to an explicit Vietnamese regulatory citation — defensible in formal safety audits."));
children.push(bullet("Swapping jurisdictions (e.g., adding Vietnamese provincial rules) is a re-ingestion task, not a re-engineering task."));
children.push(bullet("Natural fit if the PM company\u2019s end deliverable is a regulatory audit report."));
children.push(h3("Limitations"));
children.push(bullet("Retrieval quality gates the entire system — poor chunking or embedding means wrong citations."));
children.push(bullet("Higher token cost per image (retrieved clauses inflate the prompt)."));
children.push(bullet("Vietnamese-language legal text benefits from a multilingual embedding model — adds one more moving part."));

// Option 5
children.push(h2("Option 5 — Agentic VLM with Tools"));
children.push(
  p(
    "A VLM acts as a reasoning agent. When it sees an image, it can call tools: crop-and-zoom on a suspect region, invoke the specialist detector, look up a QCVN clause, or retrieve prior photos from the same project location for longitudinal context. The agent assembles a multi-violation inspection report with a full evidence trail."
  )
);
children.push(h3("Strengths"));
children.push(bullet("Best handling of complex multi-violation scenes."));
children.push(bullet("Adapts to user corrections mid-session."));
children.push(bullet("Produces a full inspection report, not just labels."));
children.push(h3("Limitations"));
children.push(bullet("Slowest (5-20 seconds per image) and most expensive per image."));
children.push(bullet("Hardest to evaluate and benchmark reliably."));
children.push(bullet("Overkill for a PM-company MVP — useful only at enterprise scale."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Infrastructure Cost Comparison ----------
children.push(h1("5. Infrastructure Cost Comparison"));
children.push(
  p(
    "All figures below are USD per month at the 8,000-photo baseline. API per-image figures assume approximately 1,800 input image-equivalent tokens plus 300 output tokens, which is typical for a construction photograph with a structured classification prompt."
  )
);

children.push(h3("5.1 Per-image inference cost (April 2026 published rates)"));
children.push(
  makeTable(
    [3200, 2080, 2080, 2000],
    [
      ["Provider / model", "Cost per image", "8,000/mo", "17,200/mo (high)"],
      ["Claude Haiku 4.5", "~ $0.003", "$ 24", "$ 52"],
      ["Claude Sonnet 4.6", "~ $0.010", "$ 80", "$ 172"],
      ["Claude Opus 4.7", "~ $0.045", "$ 360", "$ 774"],
      ["Gemini 2.x Flash", "~ $0.002", "$ 16", "$ 35"],
      ["GPT-4o", "~ $0.008", "$ 64", "$ 138"],
      ["Fine-tuned YOLO (on cloud GPU)", "~ $0.0005", "$ 4", "$ 9"],
    ],
    { header: true }
  )
);

children.push(h3("5.2 Dedicated GPU options"));
children.push(
  makeTable(
    [2400, 1600, 1600, 1600, 2160],
    [
      ["Option", "Upfront", "Monthly", "Capacity", "Notes"],
      ["RTX 4090 workstation (buy)", "~ $4,000", "~ $40", "~70k img/hr", "Power only; 3-yr amortized ≈ $150/mo"],
      ["Used A100 rig", "~ $12,000", "~ $80", "~200k img/hr", "Overkill at this volume"],
      ["Cloud RTX 4090 (Runpod, 8hr/day)", "$0", "~ $95", "~70k img/hr", "Good for periodic batch jobs"],
      ["Cloud RTX 4090 (Runpod, 24/7)", "$0", "~ $290", "~70k img/hr", "Only if latency-sensitive"],
      ["AWS g5.xlarge (A10G, 24/7)", "$0", "~ $730", "~50k img/hr", "Enterprise convenience tax"],
    ],
    { header: true }
  )
);

children.push(h3("5.3 Non-AI infrastructure (required for every option)"));
children.push(
  makeTable(
    [3200, 2080, 4080],
    [
      ["Component", "Monthly USD", "Typical provider"],
      ["Next.js frontend hosting", "~ $20", "Vercel / Fly.io"],
      ["Postgres + pgvector", "~ $25", "Supabase / Neon"],
      ["S3-compatible photo storage (24 GB new/mo, growing)", "~ $2 new + accumulating", "AWS S3 / Cloudflare R2"],
      ["Authentication (users, MFA)", "~ $25", "Clerk / Supabase Auth"],
      ["Background worker VM (queue processor)", "~ $10", "Fly.io / Hetzner"],
      ["Domain, email, monitoring", "~ $10", "Cloudflare, Sentry free tier"],
      ["Subtotal (non-AI)", "~ $92", ""],
    ],
    { header: true }
  )
);

children.push(h3("5.4 Break-even analysis: API vs. owned GPU"));
children.push(
  p(
    "An owned RTX 4090 workstation costs approximately $150/month when amortized over 36 months (hardware + electricity + rough provisioning for MLOps time). Break-even against cloud API depends on which API is being replaced:"
  )
);
children.push(bullet("vs. Claude Haiku at $0.003/image: break-even at ~50,000 images/month"));
children.push(bullet("vs. Claude Sonnet at $0.010/image: break-even at ~15,000 images/month"));
children.push(bullet("vs. Claude Opus at $0.045/image: break-even at ~3,300 images/month"));
children.push(
  p(
    "At the baseline volume of 8,000 images/month, a workstation only makes financial sense if the product would otherwise rely on Opus-tier reasoning for every image — which it does not. For the recommended Sonnet-based architecture, the crossover point is nearly twice the baseline volume. Do not purchase hardware until sustained traffic exceeds 14,000-15,000 images/month.",
    { paragraph: { spacing: { before: 120 } } }
  )
);

children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------- Total cost per option ----------
children.push(h1("6. Total Monthly Cost — All Options"));
children.push(
  p(
    "The table below combines AI inference cost with the $92/month non-AI infrastructure baseline for each pipeline option, at 8,000 photos per month. One-time setup costs (dataset labeling, custom detector training) are listed separately."
  )
);
children.push(
  makeTable(
    [2200, 1200, 1200, 1600, 3160],
    [
      ["Option", "AI/mo", "Infra/mo", "TOTAL/mo", "One-time setup"],
      ["1. Zero-shot Haiku 4.5", "$ 24", "$ 92", "~ $ 116", "~ $ 3k prompt engineering"],
      ["1. Zero-shot Sonnet 4.6 (recommended)", "$ 80", "$ 92", "~ $ 172", "~ $ 3k prompt engineering"],
      ["1. Zero-shot Gemini Flash", "$ 16", "$ 92", "~ $ 108", "~ $ 3k prompt engineering"],
      ["2. Fine-tuned detector", "$ 95-220 (GPU)", "$ 92", "~ $ 187-312", "~ $ 15-40k labeling + $ 10k training"],
      ["3. Hybrid (detector + VLM fallback)", "$ 12-25", "$ 92", "~ $ 104-117", "~ $ 15-25k (once detector built)"],
      ["4. RAG-grounded Sonnet", "~ $ 100", "$ 117 (+ vector store)", "~ $ 217", "~ $ 6k legal corpus ingestion"],
      ["5. Agentic Sonnet (multi-call)", "~ $ 240", "$ 92", "~ $ 332", "~ $ 8k tool plumbing"],
    ],
    { header: true }
  )
);

children.push(
  p(
    "Options 1 (Haiku), 1 (Gemini Flash), and 3 are clustered near the same total monthly cost (~$105-120). The differentiator among them is accuracy on Vietnamese site photography and the engineering lead time required. Zero-shot Sonnet costs 40-60% more per month but provides materially better reasoning on ambiguous multi-violation scenes, which is where inspector trust is won or lost.",
    { paragraph: { spacing: { before: 120 } } }
  )
);

// ---------- Recommendation ----------
children.push(h1("7. Recommendation"));
children.push(h3("Primary recommendation — Option 1 with Claude Sonnet 4.6"));
children.push(
  p(
    "For the MVP and the first 9-12 months of operation, deploy Option 1 (Zero-shot VLM) using Claude Sonnet 4.6 as the primary inference model, with the Vietnamese 12-category taxonomy embedded in the system prompt and QCVN 18:2021/BXD clause references cited in each response."
  )
);
children.push(h3("Why this option"));
children.push(bullet("Lowest time-to-market: a working pilot is deliverable in ~3 weeks."));
children.push(bullet("Lowest risk: no training data dependency, no ML deployment pipeline."));
children.push(bullet("Explainable: every label comes with a written rationale suitable for inspector review and for training data capture."));
children.push(bullet("Monthly cost (~$172) is trivial relative to the cost of a single safety incident on one of the ten projects."));
children.push(bullet("Vietnamese visual context (signage, PPE conventions, typical scaffolding systems) is handled reasonably by Sonnet 4.6 without fine-tuning."));
children.push(bullet("The batch-upload workflow removes any latency requirement, which is the principal weakness of VLM inference."));
children.push(bullet("Every inspector correction captured in the feedback UI becomes a labeled training example for a future Option 3 migration."));

children.push(h3("Why not the alternatives"));
children.push(
  bulletRich([
    { text: "Option 1 Haiku: ", bold: true },
    "~$56/month cheaper but noticeably weaker on ambiguous multi-worker scenes. The savings are not large enough to justify the accuracy hit when the payer is a PM company that bears liability for missed violations.",
  ])
);
children.push(
  bulletRich([
    { text: "Option 2 (fine-tuned only): ", bold: true },
    "premature. The correct label set is not yet known; committing to a detector now locks in a taxonomy before real-world data validates it.",
  ])
);
children.push(
  bulletRich([
    { text: "Option 3 (hybrid): ", bold: true },
    "the right destination, but the wrong starting point. Graduate here once 2,000+ corrected labels have accumulated.",
  ])
);
children.push(
  bulletRich([
    { text: "Option 4 (RAG-grounded): ", bold: true },
    "consider in parallel if the PM company\u2019s end deliverable is a formal audit report requiring regulatory citation. Adds ~$45/month and 2-3 weeks of engineering.",
  ])
);
children.push(
  bulletRich([
    { text: "Option 5 (agentic): ", bold: true },
    "overkill at this volume. Revisit if the product expands to enterprise-scale safety audit firms.",
  ])
);

children.push(h3("Projected monthly cost — recommended option"));
children.push(
  makeTable(
    [4000, 2360, 3000],
    [
      ["Line item", "Monthly USD", "Notes"],
      ["Claude Sonnet 4.6 inference (8k images)", "$ 80", "Baseline; scales linearly with volume"],
      ["Frontend hosting (Vercel or equivalent)", "$ 20", ""],
      ["Postgres + pgvector (Supabase/Neon)", "$ 25", "Labels, corrections, embeddings"],
      ["S3 photo storage (growing)", "$ 5", "Growing ~$2/mo each month"],
      ["Authentication (Clerk or equivalent)", "$ 25", "Per-project user isolation"],
      ["Background worker VM", "$ 10", "Queue processor for batch uploads"],
      ["Monitoring, domain, email", "$ 10", ""],
      ["TOTAL (baseline)", "~ $ 175", "Range: $ 165-210 depending on provider mix"],
    ],
    { header: true }
  )
);

// ---------- Roadmap ----------
children.push(h1("8. Implementation Roadmap"));
children.push(h3("Phase 1 — MVP (weeks 1-3)"));
children.push(bullet("Next.js + FastAPI scaffolding; S3 upload with EXIF preservation."));
children.push(bullet("Claude Sonnet 4.6 integration with VN 12-category taxonomy prompt."));
children.push(bullet("Inspector correction UI — accept / reject / reclassify per photo."));
children.push(bullet("Pilot on one of the ten projects."));
children.push(h3("Phase 2 — Hardening (months 2-4)"));
children.push(bullet("Add Vietnamese-language report export (PDF) with QCVN 18:2021/BXD clause references."));
children.push(bullet("Project-level dashboards: violation trends, repeat-offender hotspots."));
children.push(bullet("Add RAG grounding (Option 4 layer) if formal audit output is a commercial requirement."));
children.push(bullet("Extend rollout to all ten projects."));
children.push(h3("Phase 3 — Hybrid graduation (months 9-12+)"));
children.push(bullet("Labeling review gate: proceed only once 2,000+ inspector-confirmed labels exist."));
children.push(bullet("Train YOLOv8 detector on the collected dataset; deploy on a part-time cloud GPU."));
children.push(bullet("Move 80-90% of traffic to the detector; retain Sonnet for low-confidence fallback."));
children.push(bullet("Expected post-migration monthly cost: ~$110-130 with improved accuracy."));

// ---------- Risks ----------
children.push(h1("9. Risks & Open Questions"));
children.push(h3("Risks"));
children.push(
  bulletRich([
    { text: "Regulatory mis-citation: ", bold: true },
    "A wrong QCVN clause reference in a formal audit could have legal consequences. Mitigation: all Phase-1 outputs are advisory only; formal citation export requires a Vietnamese-licensed safety consultant review.",
  ])
);
children.push(
  bulletRich([
    { text: "Data sovereignty: ", bold: true },
    "Photos sent to a US-hosted VLM may conflict with project-level data protection clauses. Mitigation: confirm with each project owner; consider Gemini through a VN-region endpoint or self-host via Option 3 at Phase 3.",
  ])
);
children.push(
  bulletRich([
    { text: "Vendor pricing volatility: ", bold: true },
    "Anthropic/Google/OpenAI pricing can change. Mitigation: abstraction layer over the VLM call lets any of three providers be substituted in hours.",
  ])
);
children.push(
  bulletRich([
    { text: "Model bias on Vietnamese sites: ", bold: true },
    "VLMs are trained predominantly on Western imagery; Vietnamese bamboo scaffolding, local PPE conventions, and regional signage may be mis-read. Mitigation: the feedback loop explicitly targets these failure modes as priority training data.",
  ])
);
children.push(h3("Open questions for the PM company"));
children.push(bullet("Is the deliverable advisory (helping site supervisors prioritize) or formal (producing audit-grade reports)? This decides whether Option 4 is in scope."));
children.push(bullet("What is the expected inspector team size, and do they need offline/field capture in addition to batch upload?"));
children.push(bullet("Are the ten projects all domestic, or are any in locations where additional provincial regulations apply?"));
children.push(bullet("Is there budget willingness for a one-time $15-25k investment in labeled training data to unlock the Phase 3 cost reduction?"));

// ---------- Document ----------
const doc = new Document({
  creator: "Claude",
  title: "AI Construction Violation Classifier — Options & Recommendation",
  description: "Pipeline options, cost analysis, and recommendation for a Vietnamese construction safety violation AI webapp.",
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: INK } } },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: BRAND },
        paragraph: {
          spacing: { before: 360, after: 180 },
          outlineLevel: 0,
        },
      },
      {
        id: "Heading2",
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: BRAND },
        paragraph: {
          spacing: { before: 280, after: 140 },
          outlineLevel: 1,
        },
      },
      {
        id: "Heading3",
        name: "Heading 3",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 23, bold: true, font: "Arial", color: INK },
        paragraph: {
          spacing: { before: 220, after: 100 },
          outlineLevel: 2,
        },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: LevelFormat.BULLET,
            text: "\u2022",
            alignment: AlignmentType.LEFT,
            style: {
              paragraph: { indent: { left: 720, hanging: 360 } },
            },
          },
          {
            level: 1,
            format: LevelFormat.BULLET,
            text: "\u25E6",
            alignment: AlignmentType.LEFT,
            style: {
              paragraph: { indent: { left: 1440, hanging: 360 } },
            },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              alignment: AlignmentType.RIGHT,
              children: [
                new TextRun({
                  text: "Construction Violation AI — Options & Recommendation",
                  size: 18,
                  color: MUTED,
                  italics: true,
                }),
              ],
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: "Page ", size: 18, color: MUTED }),
                new TextRun({
                  children: [PageNumber.CURRENT],
                  size: 18,
                  color: MUTED,
                }),
                new TextRun({ text: " of ", size: 18, color: MUTED }),
                new TextRun({
                  children: [PageNumber.TOTAL_PAGES],
                  size: 18,
                  color: MUTED,
                }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
});

const outPath = path.resolve(
  "C:\\Users\\lang\\Desktop\\construct violation\\Construction_Violation_AI_Options.docx"
);
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log("Wrote " + outPath + " (" + buf.length + " bytes)");
});
