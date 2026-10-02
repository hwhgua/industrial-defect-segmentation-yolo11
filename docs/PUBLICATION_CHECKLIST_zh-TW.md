# GitHub 公開前檢查表

最後檢查日期：2026-10-02

## 已完成

- [x] 從原始約 6 GB 工作區分離出獨立乾淨 repo。
- [x] 公開主線只保留 stock Ultralytics 的 R4，不依賴 A1–A4 自訂 fork。
- [x] 修正原訓練入口無條件匯入自訂模組的問題。
- [x] 固定 R4 設定與四位數 operating point `0.0975`。
- [x] 完成 mask-to-polygon、formal split、資料驗證、訓練、mAP 評估、threshold selection、錯誤分析及推論入口。
- [x] 18 項 synthetic CPU-only 測試及 2 項 Ultralytics 整合測試通過；整合測試包含 YAML 路徑解析與一輪 CPU 訓練。
- [x] 以既有正式資料集唯讀驗證：3,335 張影像、0 個錯誤、0 個 split overlap。
- [x] 既有 `split_manifest.csv` SHA-256 符合參考值。
- [x] 修正 `data.yaml` 的 Ultralytics 路徑語意：不再寫入可能依目前工作目錄解析的 `path: .`。
- [x] 對尺寸不一致、微小前景被完全濾除及無法忠實表示的 mask 孔洞採 fail-fast。
- [x] `--overwrite` 改為先完成 staging，再替換既有資料；新建失敗時保留舊資料集。
- [x] 以 Ultralytics 8.3.221 的實際 `check_det_dataset` 驗證 synthetic `data.yaml`，train／val／test 皆解析到 YAML 所在資料集。
- [x] Python package wheel 可建立並可安裝／匯入。
- [x] 在獨立 Python 3.12 虛擬環境完成 `pip install -e ".[dev]"`，並以 stock `ultralytics==8.3.221` 跑完一輪 synthetic training、validation 與 checkpoint 存檔。
- [x] Ruff 靜態檢查通過。
- [x] 提供不含資料集影像或逐張識別資訊的彙總 CSV 與原創結果圖表。
- [x] 排除 dataset、checkpoint、cache、runs、`.ultralytics` 與本機設定。
- [x] 掃描無本機絕對路徑、使用者名稱、UUID 或疑似憑證。
- [x] 補齊 AGPL-3.0、第三方授權說明、資料來源、引用檔與 GitHub Actions。
- [x] README 明確揭露 test 歷史、validation overfitting 風險與誤報目標未達成。

## 公開前 P0

- [ ] 由 GitHub Actions 在 Python 3.10／3.11 重新驗證；本機獨立 Python 3.12 環境已完成安裝與全套 20 項測試。
- [ ] 重新取得原始 KolektorSDD2，使用本 repo 的 `ksdd2-prepare` 對真實 raw masks 跑一次端到端轉換，確認得到參考筆數與 manifest hash。目前本機只剩已轉換的 formal dataset，沒有 raw archive／masks。
- [ ] 確認 `CITATION.cff` 的作者姓名呈現方式；若要使用英文姓名，先由本人確認拼法。
- [ ] 設定 Git `user.name`／`user.email`，檢查 `git add --dry-run .` 後建立初始 commit。
- [ ] 建立 GitHub repository，先以 private 檢視 README 與 Actions；Actions 通過後再切換 public。

## 發佈選擇

- [ ] 決定是否完全不發布訓練 checkpoint（目前建議：不發布）。若要發布，需另外確認權重、Ultralytics 與資料衍生物的授權及檔案大小。
- [ ] 決定是否加入真實 TP／FN／FP 圖。目前為降低資料再散布風險而不提交。若加入，需在圖旁標示 KolektorSDD2 來源及 CC BY-NC-SA 4.0，並確認用途符合授權。
- [ ] GitHub About 建議填入：`computer-vision`, `industrial-inspection`, `semantic-segmentation`, `yolo11`, `pytorch`, `error-analysis`。

## 公開後

- [ ] 從一台未使用過此專案的環境照 README 重跑 quick start。
- [ ] 把公開 GitHub URL 加到履歷與 104 專案欄位。
- [ ] 面試前使用 `INTERVIEW_GUIDE_zh-TW.md` 練習 90 秒版本及限制題。
