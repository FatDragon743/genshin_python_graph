"""Tkinter GUI：扫码登录、同步、分析（精简版）。"""

from __future__ import annotations

import os
import tempfile
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

from .analysis import display_image_in_scroll_window, save_fig_temp
from .api import sync_wishes_to_xlsx
from .auth import clear_session, load_session, login_with_qr
from .luck import analyze_luck, format_luck_text, plot_luck_dashboard
from .xlsx_store import extract_five_star_stats, find_latest_xlsx, load_workbook


class WishAppGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("原神抽卡记录分析")
        self.root.geometry("560x420")
        self.session = load_session()
        self.fig_container = {"fig": None, "pulls": None}
        self._qr_state = None
        self.status_var = tk.StringVar()
        self.pulls_var = tk.StringVar(value="0")
        self._build()
        self._refresh_status()
        self._try_load_local_pulls()
        self._announce_cached_login()

    def _build(self):
        try:
            ttk.Style().theme_use("clam")
        except Exception:
            pass

        ttk.Label(self.root, text="原神抽卡记录分析", font=("Microsoft YaHei", 16, "bold")).pack(pady=10)

        auth = ttk.Frame(self.root)
        auth.pack(fill=tk.X, padx=16, pady=4)
        self.login_btn = ttk.Button(auth, text="米游社扫码登录", command=self._on_qr_login, width=16)
        self.login_btn.pack(side=tk.LEFT, padx=4)
        self.clear_btn = ttk.Button(auth, text="清除登录", command=self._on_clear_session, width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        actions = ttk.Frame(self.root)
        actions.pack(fill=tk.X, padx=16, pady=4)
        self.sync_btn = ttk.Button(actions, text="同步抽卡记录", command=self._on_sync, width=14)
        self.sync_btn.pack(side=tk.LEFT, padx=4)
        self.analyze_btn = ttk.Button(actions, text="欧非分析", command=self._on_analyze, width=12)
        self.analyze_btn.pack(side=tk.LEFT, padx=4)
        self.save_btn = ttk.Button(actions, text="保存图片", command=self._on_save, width=10, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(self.root, textvariable=self.status_var, font=("Microsoft YaHei", 9), foreground="#333").pack(
            anchor="w", padx=16
        )

        log_frame = ttk.LabelFrame(self.root, text="日志", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)
        self.log = scrolledtext.ScrolledText(log_frame, height=12, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True)

    def run(self):
        self.root.mainloop()

    def _append_log(self, msg: str):
        def _do():
            self.log.insert(tk.END, msg + "\n")
            self.log.see(tk.END)

        self.root.after(0, _do)

    def _refresh_status(self):
        self.session = load_session()
        logged_in = bool(self.session and self.session.is_usable())
        if logged_in:
            role = self.session.selected_role()
            self.status_var.set(
                f"已登录（本地缓存）| UID {role.game_uid} ({role.nickname})"
            )
            self.login_btn.config(text="重新扫码登录")
            self.clear_btn.config(state=tk.NORMAL)
            self.sync_btn.config(state=tk.NORMAL)
        else:
            path = find_latest_xlsx()
            extra = f" | 本地库: {path.name}" if path else ""
            self.status_var.set("未登录（分析可用本地库）" + extra)
            self.login_btn.config(text="米游社扫码登录")
            self.clear_btn.config(state=tk.DISABLED)
            self.sync_btn.config(state=tk.DISABLED)

    def _announce_cached_login(self):
        if self.session and self.session.is_usable():
            role = self.session.selected_role()
            self._append_log(
                f"已从本地缓存恢复登录 UID={role.game_uid} ({role.nickname})，可直接同步。"
            )
        else:
            self._append_log("未检测到本地登录缓存，同步前请先扫码登录。")

    def _try_load_local_pulls(self):
        try:
            path = find_latest_xlsx()
            if not path:
                return
            book = load_workbook(path)
            _, _, pulls = extract_five_star_stats(book["sheets"])
            self.pulls_var.set(str(min(pulls, 89)))
            self._append_log(f"本地库垫抽 {pulls} ← {path.name}")
        except Exception as e:
            self._append_log(f"读取本地垫抽失败: {e}")

    def _close_all_dialogs(self):
        """关闭扫码窗等所有子窗口。"""
        state = getattr(self, "_qr_state", None)
        if state:
            state["abort"] = True
        for w in list(self.root.winfo_children()):
            if isinstance(w, tk.Toplevel):
                try:
                    w.destroy()
                except Exception:
                    pass
        self._qr_state = None

    def _on_clear_session(self):
        self._close_all_dialogs()
        clear_session()
        self.session = None
        self._refresh_status()
        self._append_log("已清除登录。")

    def _on_qr_login(self):
        from .qr_login import make_qr_png

        self._close_all_dialogs()

        win = tk.Toplevel(self.root)
        win.title("米游社扫码登录")
        win.geometry("360x430")
        win.transient(self.root)
        ttk.Label(
            win,
            text="米游社 APP → 我的 → 扫一扫",
            font=("Microsoft YaHei", 10),
        ).pack(pady=8)
        img_label = ttk.Label(win)
        img_label.pack(pady=4)
        status_var = tk.StringVar(value="正在生成二维码…")
        ttk.Label(win, textvariable=status_var, wraplength=320).pack(pady=4)

        state = {"abort": False, "win": win}
        self._qr_state = state
        photo_holder = {"img": None}

        def finish_win():
            try:
                if win.winfo_exists():
                    win.destroy()
            except Exception:
                pass
            if getattr(self, "_qr_state", None) is state:
                self._qr_state = None

        def on_close():
            state["abort"] = True
            finish_win()

        win.protocol("WM_DELETE_WINDOW", on_close)
        ttk.Button(win, text="取消", command=on_close).pack(pady=8)

        def work():
            try:
                def on_qr(url: str):
                    if state["abort"]:
                        return
                    path = os.path.join(tempfile.gettempdir(), "mihoyo_qr_login.png")
                    make_qr_png(url, path)

                    def show():
                        if state["abort"] or not win.winfo_exists():
                            return
                        try:
                            from PIL import Image, ImageTk

                            photo = ImageTk.PhotoImage(Image.open(path).resize((280, 280)))
                            photo_holder["img"] = photo
                            img_label.configure(image=photo)
                            status_var.set("请扫码…")
                        except Exception as ex:
                            status_var.set(str(ex))

                    self.root.after(0, show)

                def on_status(msg: str):
                    if state["abort"]:
                        return
                    self._append_log(msg)
                    self.root.after(
                        0,
                        lambda m=msg: status_var.set(m) if win.winfo_exists() else None,
                    )

                session = login_with_qr(
                    timeout=180,
                    on_qr=on_qr,
                    on_status=on_status,
                    should_abort=lambda: state["abort"],
                )
                if state["abort"]:
                    return
                self.session = session
                role = session.selected_role()
                err = None

                def done():
                    finish_win()
                    self._append_log(f"登录成功 UID={role.game_uid} {role.nickname}")
                    self._refresh_status()
                    messagebox.showinfo("成功", "登录完成，请点击「同步抽卡记录」。")

                self.root.after(0, done)
            except Exception as e:
                err_msg = str(e) or repr(e)
                if state["abort"]:
                    return
                self._append_log(f"扫码失败: {err_msg}")

                def fail(msg=err_msg):
                    finish_win()
                    messagebox.showerror("扫码失败", msg)

                self.root.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def _on_sync(self):
        self._close_all_dialogs()
        self.session = load_session()
        if not self.session or not self.session.is_usable():
            messagebox.showwarning("未登录", "请先扫码登录。")
            return
        self.analyze_btn.config(state=tk.DISABLED)

        def work():
            try:
                saved, report, book = sync_wishes_to_xlsx(
                    self.session,
                    on_progress=self._append_log,
                )
                _, _, pulls = extract_five_star_stats(book["sheets"])
                shown = min(pulls, 89)
                summary = report.summary()
                name = saved.name

                def done():
                    self.pulls_var.set(str(shown))
                    self._append_log(f"同步完成: {name}")
                    self._append_log(summary)
                    self._refresh_status()
                    self.analyze_btn.config(state=tk.NORMAL)
                    messagebox.showinfo("同步完成", f"{name}\n{summary}")

                self.root.after(0, done)
            except Exception as e:
                err_msg = str(e) or repr(e)
                self._append_log(f"同步失败: {err_msg}")

                def fail(msg=err_msg):
                    self.analyze_btn.config(state=tk.NORMAL)
                    messagebox.showerror("同步失败", msg)

                self.root.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def _on_analyze(self):
        self._close_all_dialogs()
        self.analyze_btn.config(state=tk.DISABLED)

        def work():
            try:
                path = find_latest_xlsx()
                if path is None:
                    raise RuntimeError("没有本地库，请先同步抽卡记录。")
                book = load_workbook(path)
                uid = str((book.get("meta") or {}).get("uid") or "")
                report = analyze_luck(book["raw"], source=path.name, uid=uid)
                text = format_luck_text(report)
                print(text)
                fig = plot_luck_dashboard(report)
                tmp = save_fig_temp(fig)
                char = next((p for p in report.pools if p.key == "character"), None)
                pity = char.current_pity if char else 0

                def done():
                    self.fig_container["fig"] = fig
                    self.fig_container["pulls"] = pity
                    self.pulls_var.set(str(pity))
                    self.save_btn.config(state=tk.NORMAL)
                    self.analyze_btn.config(state=tk.NORMAL)
                    self._append_log(text)
                    try:
                        display_image_in_scroll_window(tmp)
                    except Exception:
                        pass

                self.root.after(0, done)
            except Exception as e:
                err_msg = str(e) or repr(e)
                self._append_log(f"分析失败: {err_msg}")

                def fail(msg=err_msg):
                    self.analyze_btn.config(state=tk.NORMAL)
                    messagebox.showerror("分析失败", msg)

                self.root.after(0, fail)

        threading.Thread(target=work, daemon=True).start()

    def _on_save(self):
        fig = self.fig_container.get("fig")
        if fig is None:
            messagebox.showwarning("提示", "请先分析。")
            return
        try:
            save_dir = os.path.expanduser("~/Pictures/GenshinWishAnalysis")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(
                save_dir,
                f"luck_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            fig.savefig(filepath, dpi=200, bbox_inches="tight")
            messagebox.showinfo("已保存", filepath)
        except Exception as e:
            messagebox.showerror("保存失败", str(e) or repr(e))


def create_gui():
    WishAppGUI().run()


def run():
    create_gui()
