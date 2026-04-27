"""
App com drag-and-drop para processar imagens WhatsApp e exportar para Excel.
Correr: conda activate whatsapp_to_excel && python src/app.py
"""
import hashlib
import pickle
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, ttk

import customtkinter as ctk
import openpyxl
from tkinterdnd2 import DND_FILES, TkinterDnD

sys.path.insert(0, str(Path(__file__).parent))

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_PATH = DATA_DIR / "cache.pkl"
OUTPUT_DIR.mkdir(exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

IMG_EXTS = {".png", ".jpg", ".jpeg"}


# =========================
# CACHE
# =========================
def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


# =========================
# APP
# =========================
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhatsApp → Excel")
        self.geometry("1080x660")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)

        self._resultados: dict[str, list] = {}   # {img_path: [(produto, score, qty)]}
        self._cache: dict       = _load_cache()
        self._pipeline          = None            # carregado em background
        self._img_labels: dict  = {}              # nome → CTkLabel (para actualizar ícone)

        self._build_ui()

        # Registar DnD na janela raiz — no macOS CTkFrame não propaga drops
        self.drop_target_register(DND_FILES)
        self.dnd_bind("<<Drop>>", self._on_drop)

        self._status("A carregar modelos OCR + NLP…", loading=True)
        threading.Thread(target=self._load_pipeline, daemon=True).start()

    # ── PIPELINE LOAD ─────────────────────────────────────────────
    def _load_pipeline(self):
        import pipeline as pl
        produtos, emb_prod, sku_map = pl.load_produtos()
        self._pipeline = {"pl": pl, "produtos": produtos, "emb_prod": emb_prod, "sku_map": sku_map}
        self.after(0, lambda: self._status("Pronto — arrasta as imagens ou clica na zona de drop", ok=True))

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # top bar
        bar = ctk.CTkFrame(self, height=48, fg_color="#16213e")
        bar.pack(fill="x", side="top")
        ctk.CTkLabel(bar, text="WhatsApp → Excel",
                     font=("Helvetica", 16, "bold"), text_color="#4fc3f7").pack(side="left", padx=16, pady=10)

        # main area
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=12, pady=(6, 0))

        # ── left panel ──
        left = ctk.CTkFrame(main, width=285, fg_color="#16213e", corner_radius=10)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        # drop zone
        self._drop_frame = ctk.CTkFrame(left, height=130, fg_color="#0f3460",
                                         corner_radius=10, border_width=2,
                                         border_color="#4fc3f7")
        self._drop_frame.pack(fill="x", padx=12, pady=(12, 6))
        self._drop_frame.pack_propagate(False)
        self._drop_icon = ctk.CTkLabel(self._drop_frame, text="📂",
                                        font=("", 32))
        self._drop_icon.pack(pady=(16, 2))
        drop_txt = ctk.CTkLabel(self._drop_frame,
                                text="Arrasta as imagens\nou clica para seleccionar",
                                font=("Helvetica", 11), text_color="#90caf9", justify="center")
        drop_txt.pack()

        for w in (self._drop_frame, self._drop_icon, drop_txt):
            w.bind("<Button-1>", lambda e: self._browse())
            w.bind("<Enter>", lambda e: self._drop_frame.configure(border_color="#81d4fa"))
            w.bind("<Leave>", lambda e: self._drop_frame.configure(border_color="#4fc3f7"))

        ctk.CTkLabel(left, text="Imagens", font=("Helvetica", 12, "bold"),
                     text_color="#e0e0e0").pack(anchor="w", padx=14, pady=(8, 2))

        self._list_frame = ctk.CTkScrollableFrame(left, fg_color="#0f3460", corner_radius=8)
        self._list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # ── right panel ──
        right = ctk.CTkFrame(main, fg_color="#16213e", corner_radius=10)
        right.pack(side="left", fill="both", expand=True)

        ctk.CTkLabel(right, text="Resultados", font=("Helvetica", 12, "bold"),
                     text_color="#e0e0e0").pack(anchor="w", padx=14, pady=(10, 4))

        tree_frame = ctk.CTkFrame(right, fg_color="transparent")
        tree_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.Treeview",
                        background="#0f3460", foreground="#e0e0e0",
                        fieldbackground="#0f3460", rowheight=26,
                        font=("Helvetica", 11))
        style.configure("App.Treeview.Heading",
                        background="#16213e", foreground="#4fc3f7",
                        font=("Helvetica", 11, "bold"), relief="flat")
        style.map("App.Treeview", background=[("selected", "#1565c0")])

        self._tree = ttk.Treeview(
            tree_frame,
            columns=("produto", "qtd", "score", "imagem"),
            show="headings", style="App.Treeview"
        )
        self._tree.heading("produto", text="Produto")
        self._tree.heading("qtd",     text="Qtd")
        self._tree.heading("score",   text="Score")
        self._tree.heading("imagem",  text="Imagem")
        self._tree.column("produto",  width=420, stretch=True)
        self._tree.column("qtd",      width=50,  anchor="center", stretch=False)
        self._tree.column("score",    width=70,  anchor="center", stretch=False)
        self._tree.column("imagem",   width=190, stretch=False)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # ── bottom bar ──
        bottom = ctk.CTkFrame(self, height=50, fg_color="#16213e")
        bottom.pack(fill="x", side="bottom")

        ctk.CTkButton(bottom, text="Exportar Excel", width=140,
                      fg_color="#1565c0", hover_color="#1976d2",
                      command=self._exportar).pack(side="left", padx=12, pady=8)
        ctk.CTkButton(bottom, text="Limpar", width=80,
                      fg_color="#37474f", hover_color="#546e7a",
                      command=self._limpar).pack(side="left", padx=(0, 8), pady=8)
        ctk.CTkButton(bottom, text="Limpar cache", width=110,
                      fg_color="#37474f", hover_color="#546e7a",
                      command=self._limpar_cache).pack(side="left", pady=8)
        ctk.CTkButton(bottom, text="Colar texto", width=100,
                      fg_color="#1b5e20", hover_color="#2e7d32",
                      command=self._abrir_colar_texto).pack(side="left", padx=(8, 0), pady=8)

        self._progress = ctk.CTkProgressBar(bottom, width=120, mode="indeterminate")
        self._progress.pack(side="right", padx=(0, 8), pady=14)
        self._progress.start()

        self._status_lbl = ctk.CTkLabel(bottom, text="A carregar…",
                                         font=("Helvetica", 11), text_color="#90caf9")
        self._status_lbl.pack(side="right", padx=(0, 4))

    # ── STATUS BAR ────────────────────────────────────────────────
    def _status(self, msg: str, loading=False, ok=False, error=False):
        color = "#4fc3f7" if loading else ("#66bb6a" if ok else ("#ef5350" if error else "#e0e0e0"))
        self._status_lbl.configure(text=msg, text_color=color)
        if loading:
            self._progress.pack(side="right", padx=(0, 8), pady=14)
            self._progress.start()
        else:
            self._progress.stop()
            self._progress.pack_forget()

    # ── DROP / BROWSE ─────────────────────────────────────────────
    def _browse(self):
        if not self._pipeline:
            self._status("Aguarda — modelos ainda a carregar…", loading=True)
            return
        paths = filedialog.askopenfilenames(
            title="Seleccionar imagens",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg")]
        )
        if paths:
            threading.Thread(target=self._processar, args=(list(paths),), daemon=True).start()

    def _on_drop(self, event):
        if not self._pipeline:
            self._status("Aguarda — modelos ainda a carregar…", loading=True)
            return

        # macOS: paths com espaços chegam entre {}; usar splitlist do Tcl para parsing seguro
        try:
            raw_paths = self.tk.splitlist(event.data)
        except Exception:
            raw_paths = event.data.split()
        paths = [p.strip("{}") for p in raw_paths]
        imgs  = [p for p in paths if Path(p).suffix.lower() in IMG_EXTS]

        if not imgs:
            self._status("Nenhuma imagem válida (aceita .png .jpg .jpeg)", error=True)
            return

        threading.Thread(target=self._processar, args=(imgs,), daemon=True).start()

    # ── PROCESSAMENTO ─────────────────────────────────────────────
    def _set_img_estado(self, nome: str, estado: str):
        """Adiciona ou actualiza a linha do ficheiro na lista da esquerda."""
        icon  = {"ok": "✓", "loading": "⧗", "cache": "⚡", "error": "✕"}.get(estado, "?")
        color = {"ok": "#66bb6a", "loading": "#ffa726",
                 "cache": "#4fc3f7", "error": "#ef5350"}.get(estado, "#e0e0e0")
        label_text = f"{icon}  {nome[:32]}"

        if nome in self._img_labels:
            self._img_labels[nome].configure(text=label_text, text_color=color)
        else:
            lbl = ctk.CTkLabel(self._list_frame, text=label_text,
                               text_color=color, anchor="w", font=("Helvetica", 11))
            lbl.pack(fill="x", padx=4, pady=2)
            self._img_labels[nome] = lbl

    def _processar(self, paths: list[str]):
        self.after(0, lambda: self._status(f"A processar {len(paths)} imagem(ns)…", loading=True))

        for path in paths:
            nome = Path(path).name
            key  = _md5(path)
            self.after(0, self._set_img_estado, nome, "loading")

            if key in self._cache:
                resultado = self._cache[key]
                self.after(0, self._set_img_estado, nome, "cache")
            else:
                try:
                    pl       = self._pipeline
                    resultado = pl["pl"].processar_imagem(Path(path), pl["produtos"], pl["emb_prod"])
                    self._cache[key] = resultado
                    _save_cache(self._cache)
                    self.after(0, self._set_img_estado, nome, "ok")
                except Exception as exc:
                    self.after(0, self._set_img_estado, nome, "error")
                    print(f"[ERRO] {nome}: {exc}")
                    continue

            self._resultados[path] = resultado
            self.after(0, self._add_rows, nome, resultado)

        total = sum(len(v) for v in self._resultados.values())
        self.after(0, lambda: self._status(f"{total} produto(s) detectado(s)", ok=True))

    def _add_rows(self, img_nome: str, resultado: list):
        for produto, score, qty in resultado:
            tag = "high" if score >= 0.95 else ("mid" if score >= 0.85 else "low")
            self._tree.insert("", "end",
                              values=(produto, qty, f"{score:.3f}", img_nome),
                              tags=(tag,))
        self._tree.tag_configure("high", foreground="#66bb6a")
        self._tree.tag_configure("mid",  foreground="#ffa726")
        self._tree.tag_configure("low",  foreground="#ef5350")

    # ── EXPORTAR ──────────────────────────────────────────────────
    def _exportar(self):
        if not self._resultados:
            self._status("Sem resultados para exportar", error=True)
            return

        path = filedialog.asksaveasfilename(
            title="Guardar Excel",
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            initialfile="resultados.xlsx"
        )
        if not path:
            return

        sku_map = self._pipeline.get("sku_map", {}) if self._pipeline else {}
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Resultados"
        ws.append(["Imagem", "Referência", "Produto", "Quantidade", "Score"])
        for cell in ws[1]:
            cell.font = openpyxl.styles.Font(bold=True)
        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 15
        ws.column_dimensions["C"].width = 55
        ws.column_dimensions["D"].width = 12
        ws.column_dimensions["E"].width = 10

        for img_path, produtos in self._resultados.items():
            nome = Path(img_path).name
            for p, s, qty in produtos:
                ref = sku_map.get(p, "")
                ws.append([nome, ref, p, qty, round(s, 4)])

        wb.save(path)
        self._status("Excel exportado ✓", ok=True)
        subprocess.run(["open", path])

    # ── LIMPAR ────────────────────────────────────────────────────
    def _limpar(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for child in list(self._list_frame.winfo_children()):
            child.destroy()
        self._img_labels.clear()
        self._resultados.clear()
        self._status("Pronto", ok=True)

    def _limpar_cache(self):
        self._cache.clear()
        _save_cache(self._cache)
        self._status("Cache limpa ✓", ok=True)

    # ── COLAR TEXTO ───────────────────────────────────────────────
    def _abrir_colar_texto(self):
        if not self._pipeline:
            self._status("Aguarda — modelos ainda a carregar…", loading=True)
            return

        win = ctk.CTkToplevel(self)
        win.title("Colar texto WhatsApp")
        win.geometry("500x380")
        win.configure(fg_color="#1a1a2e")
        win.grab_set()

        ctk.CTkLabel(win, text="Cola aqui o texto da encomenda:",
                     font=("Helvetica", 12), text_color="#90caf9").pack(anchor="w", padx=16, pady=(14, 4))

        caixa = ctk.CTkTextbox(win, font=("Helvetica", 12),
                               fg_color="#0f3460", text_color="#e0e0e0",
                               corner_radius=8)
        caixa.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        caixa.focus()

        def processar():
            texto = caixa.get("1.0", "end").strip()
            if not texto:
                return
            win.destroy()
            threading.Thread(target=self._processar_texto_colado,
                             args=(texto,), daemon=True).start()

        ctk.CTkButton(win, text="Processar", fg_color="#1565c0",
                      hover_color="#1976d2", command=processar).pack(pady=(0, 14))

    def _processar_texto_colado(self, texto: str):
        self.after(0, lambda: self._status("A processar texto…", loading=True))
        try:
            pl       = self._pipeline
            resultado = pl["pl"].processar_texto(texto, pl["produtos"], pl["emb_prod"])
            nome = "texto colado"
            self._resultados[nome] = resultado
            self.after(0, self._add_rows, nome, resultado)
            total = sum(len(v) for v in self._resultados.values())
            self.after(0, lambda: self._status(f"{total} produto(s) detectado(s)", ok=True))
        except Exception as exc:
            self.after(0, lambda: self._status(f"Erro: {exc}", error=True))
            print(f"[ERRO texto] {exc}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app = App()
    app.mainloop()
