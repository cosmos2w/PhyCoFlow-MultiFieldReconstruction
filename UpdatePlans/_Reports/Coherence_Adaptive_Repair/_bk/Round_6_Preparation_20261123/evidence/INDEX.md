# Historical archive status

This evidence index records the preparation-close publication as it stood on 2026-11-23. A later user-authorized resumption superseded that disposition. The original report receipt and evidence hashes below refer to the historical source publication, not the archive presentation changes.

# Round 6 preparation-close evidence

The selected publication is an ordinary-file copy of the original DATA2 campaign report `reports/preparation_close_v2/`. The original report retains its input manifest, complete evidence index, HTML and all nine page previews on DATA2. Raw TRAIN800 rows and checkpoint/config bindings remain campaign-owned and are identified by the compact decision's evidence records; local DATA2 paths are provenance references, not portable links.

- [Joint audit decision](joint_audit_decision.json): completed, decision `unqualified_pi_retain_source`; SHA-256 `db391c44e8976d950a07dcef22680c80af5152d96412fba70d8194013c6ecd31`.
- [Engineering decision](engineering_decision.json): retain existing execution; SHA-256 `82a9b61e667a20d710eb538bcecc3bcbb9b52dc368ee6456ac0db36ee76599a4`.
- [Preparation cost](cost_summary.json): closed lease ledger and measurement boundaries; SHA-256 `7573d53b11d3c6e7982712ecb03e5961b3df454cc4e316c96f83e42afb86b0ed`.
- [Report build receipt](report_receipt.json): 9 pages, 10 figure PNG/PDF pairs, 7.188 seconds wall and 7.186 seconds user plus system CPU, separate from occupied GPU time. All nine rendered pages were independently inspected after generation; the original build receipt retains its pre-inspection status.

The report was generated from `configs/preparation_report_inputs.json` in the campaign, SHA-256 `a8ad0ae54b27816e2c7c728f8dce7079ac16a6cf06b49e254f7c6e43b5bb82f7`. Its complete figure and report hashes are in the report build receipt. The historical Round 3 history is a 500-epoch reference, not a Round 6 learning experiment. The report generation used no new inference or model assessment.

The published Markdown and figures retain the original bytes. The published main PDF changes only 14 figure-master link annotations to relative `figures/` references, so the links continue to resolve after relocation; all nine rendered page pixels match the original. Published PDF SHA-256 is `b18b089f85b84a60b334a56aaf404178003cc5c1a4e98e4dc5ea771a06220e71`; the original PDF hash remains in the build receipt.

The primary reviewer inspected all nine v2 page previews: numerical tables, captions and all ten figure panels are readable and uncropped. The final publication passes the portable collection check. Final clean Git export, PDF-link and file-hash checks are recorded in the delivery audit.

[Delivery audit](DELIVERY_AUDIT.md) records independent inspection, clean Git export, focused tests, source capture and scope limits.
