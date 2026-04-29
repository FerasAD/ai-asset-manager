import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog,
    QMessageBox, QLineEdit, QComboBox, QTextEdit,
    QListWidgetItem, QFrame,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtCore import Qt, QUrl, QThread, Signal

from services.file_scanner import scan_audio_files
from services.audio_metadata import get_audio_metadata
from services.audio_tagger import generate_filename_tags
from services.audio_embedder import embed_audio_file
from services.semantic_search import (
    build_asset_text,
    semantic_search_audio,
    semantic_search_text_fallback,
)
from database.asset_repository import (
    insert_asset, insert_tag,
    get_tags_for_asset, get_all_assets,
    get_all_assets_with_tags, search_assets,
    log_search, store_audio_embedding,
    get_all_audio_embeddings, has_audio_embedding,
)


# ── Background worker ─────────────────────────────────────────────────────────

class EmbeddingWorker(QThread):
    # Runs CLAP embedding in a separate thread so the UI doesn't freeze during import
    progress = Signal(str)       # emits a status message per file → updates status bar
    finished = Signal(int, int)  # emits (success_count, fail_count) when done

    def __init__(self, file_asset_pairs: list):
        super().__init__()
        self.file_asset_pairs = file_asset_pairs  # list of (file_path, asset_id) tuples

    def run(self):
        # Loops through every file, skips already-embedded ones, stores new embeddings
        success = 0
        fail = 0
        total = len(self.file_asset_pairs)

        for i, (file_path, asset_id) in enumerate(self.file_asset_pairs, start=1):
            filename = os.path.basename(file_path)

            if has_audio_embedding(asset_id):
                success += 1
                self.progress.emit(f"[{i}/{total}] Already embedded: {filename}")
                continue

            self.progress.emit(f"[{i}/{total}] Embedding: {filename}")
            embedding = embed_audio_file(file_path)

            if embedding is not None:
                store_audio_embedding(asset_id, embedding)
                success += 1
            else:
                fail += 1

        self.finished.emit(success, fail)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Asset Manager")
        self.setMinimumSize(1150, 720)

        # Instance variables used across methods
        self.selected_folder = None
        self.audio_files = []
        self.current_assets = []      # assets currently shown in the list
        self.score_lookup = {}        # maps asset_id → semantic score for display
        self.embedding_worker = None  # reference to the background thread

        # Audio playback setup
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.7)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.apply_styles()

        # Root layout — stacks header, controls, content, and status bar vertically
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        central_widget.setLayout(main_layout)

        # ── Header ────────────────────────────────────────────────────────────
        header_container = QFrame()
        header_container.setObjectName("card")
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(16, 16, 16, 16)
        header_layout.setSpacing(8)

        self.title_label = QLabel("AI-Powered Audio Asset Manager")
        self.title_label.setObjectName("titleLabel")

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setObjectName("subtitleLabel")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.folder_label)
        header_container.setLayout(header_layout)

        # ── Controls ──────────────────────────────────────────────────────────
        controls_container = QFrame()
        controls_container.setObjectName("card")
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(16, 16, 16, 16)
        controls_layout.setSpacing(10)

        self.select_folder_button = QPushButton("Select Audio Folder")
        self.select_folder_button.clicked.connect(self.select_folder)

        # Dropdown to switch between keyword and semantic search
        self.search_mode = QComboBox()
        self.search_mode.addItems(["Keyword Search", "Semantic Search"])
        self.search_mode.currentTextChanged.connect(self.filter_assets)

        # Fires filter_assets on every keystroke
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search by keyword or meaning...")
        self.search_bar.textChanged.connect(self.filter_assets)

        # Toggle to hide low-relevance semantic results below score threshold
        self.filter_toggle = QPushButton("Filter: Off")
        self.filter_toggle.setCheckable(True)
        self.filter_toggle.setChecked(False)
        self.filter_toggle.clicked.connect(self._on_filter_toggle)

        controls_layout.addWidget(self.select_folder_button, 1)
        controls_layout.addWidget(self.search_mode, 1)
        controls_layout.addWidget(self.search_bar, 3)
        controls_layout.addWidget(self.filter_toggle, 1)
        controls_container.setLayout(controls_layout)

        # ── Main content — left list + right details ───────────────────────── 
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        left_panel = QFrame()
        left_panel.setObjectName("card")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(10)

        self.results_label = QLabel("Results")
        self.results_label.setObjectName("sectionLabel")

        # Selecting a row triggers show_asset_details
        self.audio_list = QListWidget()
        self.audio_list.currentRowChanged.connect(self.show_asset_details)

        left_layout.addWidget(self.results_label)
        left_layout.addWidget(self.audio_list)
        left_panel.setLayout(left_layout)

        right_panel = QFrame()
        right_panel.setObjectName("card")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)

        self.details_label = QLabel("Asset Details")
        self.details_label.setObjectName("sectionLabel")

        self.details_panel = QTextEdit()
        self.details_panel.setReadOnly(True)

        self.play_button = QPushButton("Play Selected Audio")
        self.play_button.clicked.connect(self.play_selected_audio)

        self.stop_button = QPushButton("Stop Audio")
        self.stop_button.clicked.connect(self.stop_audio)

        right_layout.addWidget(self.details_label)
        right_layout.addWidget(self.details_panel)
        right_layout.addWidget(self.play_button)
        right_layout.addWidget(self.stop_button)
        right_panel.setLayout(right_layout)

        content_layout.addWidget(left_panel, 2)
        content_layout.addWidget(right_panel, 1)

        # ── Status bar ────────────────────────────────────────────────────────
        status_container = QFrame()
        status_container.setObjectName("card")
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(16, 10, 16, 10)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.status_label.setObjectName("statusLabel")

        status_layout.addWidget(self.status_label)
        status_container.setLayout(status_layout)

        main_layout.addWidget(header_container)
        main_layout.addWidget(controls_container)
        main_layout.addLayout(content_layout)
        main_layout.addWidget(status_container)

        # Load any previously imported assets from the database on startup
        self.load_assets_from_database()

    def apply_styles(self):
        # All CSS-like styling for the UI — colours, fonts, hover states, border radius
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f6f8fb;
                color: #1f2937;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 13px;
            }
            QFrame#card {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
            }
            QLabel#titleLabel {
                font-size: 24px;
                font-weight: 700;
                color: #111827;
                background: transparent;
            }
            QLabel#subtitleLabel {
                color: #4b5563;
                font-size: 13px;
                background: transparent;
            }
            QLabel#sectionLabel {
                font-size: 15px;
                font-weight: 600;
                color: #111827;
                background: transparent;
            }
            QLabel#statusLabel {
                color: #475569;
                font-size: 12px;
                background: transparent;
            }
            QPushButton {
                background-color: #dbeafe;
                color: #1d4ed8;
                border: 1px solid #bfdbfe;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #bfdbfe; }
            QPushButton:pressed { background-color: #93c5fd; }
            QPushButton:disabled {
                background-color: #f3f4f6;
                color: #9ca3af;
                border: 1px solid #e5e7eb;
            }
            QPushButton#filterActive {
                background-color: #1d4ed8;
                color: #ffffff;
                border: 1px solid #1e40af;
            }
            QPushButton#filterActive:hover { background-color: #1e40af; }
            QLineEdit, QComboBox, QTextEdit, QListWidget {
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 10px;
                padding: 8px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus, QListWidget:focus {
                border: 1px solid #60a5fa;
            }
            QListWidget { padding: 6px; }
            QListWidget::item {
                padding: 8px;
                border-radius: 8px;
                margin-bottom: 2px;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
                color: #111827;
            }
            QListWidget::item:hover { background-color: #eff6ff; }
            QComboBox { padding-right: 24px; }
        """)

    def select_folder(self):
        # Opens a folder picker dialog, updates the header label, triggers file loading
        folder = QFileDialog.getExistingDirectory(self, "Select Audio Folder")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(f"Selected folder: {folder}")
            self.load_audio_files()

    def load_audio_files(self):
        # Full import pipeline: scan → metadata → insert → tag → start embedding worker
        self.audio_files = scan_audio_files(self.selected_folder)

        if not self.audio_files:
            QMessageBox.information(
                self, "No Audio Files Found",
                "No supported audio files were found in this folder.",
            )
            return

        file_asset_pairs = []

        for file_path in self.audio_files:
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()
            duration, file_size_mb = get_audio_metadata(file_path)

            asset_id = insert_asset(file_name, file_path, file_extension, duration, file_size_mb)

            if asset_id is not None:
                tags = generate_filename_tags(file_path)
                for tag in tags:
                    insert_tag(asset_id, tag, source="filename_auto")
                file_asset_pairs.append((file_path, asset_id))

        self.status_label.setText(
            f"Imported {len(self.audio_files)} file(s). Starting audio embedding..."
        )
        self.load_assets_from_database()
        self._start_embedding_worker(file_asset_pairs)

    def _start_embedding_worker(self, file_asset_pairs: list):
        # Spawns EmbeddingWorker thread — disables folder button until embedding finishes
        if not file_asset_pairs:
            return

        self.select_folder_button.setEnabled(False)

        self.embedding_worker = EmbeddingWorker(file_asset_pairs)
        self.embedding_worker.progress.connect(self._on_embedding_progress)
        self.embedding_worker.finished.connect(self._on_embedding_finished)
        self.embedding_worker.start()

    def _on_embedding_progress(self, message: str):
        # Updates status bar with each file's embedding progress
        self.status_label.setText(message)

    def _on_embedding_finished(self, success: int, fail: int):
        # Re-enables folder button and shows final embedding summary
        self.select_folder_button.setEnabled(True)
        msg = f"Embedding complete — {success} embedded"
        if fail:
            msg += f", {fail} failed (will use text fallback)"
        self.status_label.setText(msg)

    def load_assets_from_database(self):
        # Fetches all assets from the database and renders them in the list
        assets = get_all_assets()
        self.display_assets(assets)

    def _on_filter_toggle(self):
        # Updates button label and style, then re-runs the current search to apply or remove filter
        if self.filter_toggle.isChecked():
            self.filter_toggle.setText("Filter: On")
            self.filter_toggle.setObjectName("filterActive")
        else:
            self.filter_toggle.setText("Filter: Off")
            self.filter_toggle.setObjectName("")
        self.filter_toggle.setStyle(self.filter_toggle.style())
        self.filter_assets()

    def filter_assets(self):
        # Called on every keystroke — routes to keyword or semantic search based on dropdown
        query = self.search_bar.text().strip()

        if query == "":
            assets = get_all_assets()
            self.display_assets(assets)
            return

        if self.search_mode.currentText() == "Keyword Search":
            assets = search_assets(query)
            self.display_assets(assets)
            log_search(query, "keyword", len(assets))
            self.status_label.setText(
                f"Keyword search: {len(assets)} result(s) for '{query}'."
            )

        else:
            # Primary path: CLAP audio embeddings
            embedded_assets = get_all_audio_embeddings()

            if embedded_assets:
                results = semantic_search_audio(query, embedded_assets, top_k=20)

                # If filter is active, remove results below the relevance threshold
                if self.filter_toggle.isChecked():
                    results = [r for r in results if r["score"] >= 0.25]

                ranked_assets = [r["asset"] for r in results]
                self.display_assets(ranked_assets, semantic_scores=results)
                log_search(query, "semantic_audio", len(ranked_assets))

                filter_note = " (filtered)" if self.filter_toggle.isChecked() else ""
                self.status_label.setText(
                    f"Audio semantic search{filter_note}: {len(ranked_assets)} result(s) for '{query}'."
                )

            else:
                # Fallback: no embeddings yet, use filename+tag text similarity instead
                assets_with_tags = get_all_assets_with_tags()
                semantic_input = [
                    {
                        "asset": item["asset"],
                        "text": build_asset_text(item["asset"][1], item["tags"])
                    }
                    for item in assets_with_tags
                ]
                results = semantic_search_text_fallback(query, semantic_input, top_k=20)
                ranked_assets = [r["asset"] for r in results]
                self.display_assets(ranked_assets, semantic_scores=results)
                log_search(query, "semantic_text_fallback", len(ranked_assets))
                self.status_label.setText(
                    f"Text semantic search (audio embeddings still processing): "
                    f"{len(ranked_assets)} result(s) for '{query}'."
                )

    def display_assets(self, assets, semantic_scores=None):
        # Clears and repopulates the left panel list with the given assets
        # If semantic scores are provided, appends the score to each list item
        self.audio_list.clear()
        self.details_panel.clear()
        self.current_assets = assets

        self.score_lookup = {}
        if semantic_scores:
            for result in semantic_scores:
                asset_id = result["asset"][0]
                self.score_lookup[asset_id] = result["score"]

        for asset in assets:
            asset_id, filename, filepath, filetype, duration, file_size_mb, imported_at = asset
            tags = get_tags_for_asset(asset_id)
            tags_text = ", ".join(tags) if tags else "no tags"
            duration_text = f"{duration}s" if duration is not None else "Unknown duration"
            item_text = f"{filename} | {duration_text} | {tags_text}"

            if asset_id in self.score_lookup:
                item_text += f" | Score: {self.score_lookup[asset_id]:.3f}"

            self.audio_list.addItem(QListWidgetItem(item_text))

        self.results_label.setText(f"Results ({len(assets)})")

        if assets:
            self.audio_list.setCurrentRow(0)
        else:
            self.details_panel.setPlainText("No asset selected.")
            if self.search_bar.text().strip():
                self.status_label.setText("No matching results found.")
            else:
                self.status_label.setText("No assets loaded.")

    def show_asset_details(self, row):
        # Populates the right panel with full metadata for the selected asset
        if row < 0 or row >= len(self.current_assets):
            self.details_panel.setPlainText("No asset selected.")
            return

        asset = self.current_assets[row]
        asset_id, filename, filepath, filetype, duration, file_size_mb, imported_at = asset
        tags = get_tags_for_asset(asset_id)

        duration_text = f"{duration}s" if duration is not None else "Unknown duration"
        size_text = f"{file_size_mb} MB" if file_size_mb is not None else "Unknown size"
        tags_text = ", ".join(tags) if tags else "no tags"

        embedded_status = "Yes" if has_audio_embedding(asset_id) else "No (pending or failed)"

        details_lines = [
            f"Filename: {filename}",
            f"File type: {filetype}",
            f"Duration: {duration_text}",
            f"Size: {size_text}",
            f"Imported at: {imported_at}",
            f"Path: {filepath}",
            f"Tags: {tags_text}",
            f"Audio embedded: {embedded_status}",
        ]

        if asset_id in self.score_lookup:
            details_lines.append(f"Semantic score: {self.score_lookup[asset_id]:.3f}")

        self.details_panel.setPlainText("\n".join(details_lines))

    def play_selected_audio(self):
        # Gets the filepath of the selected asset and plays it via QMediaPlayer
        current_row = self.audio_list.currentRow()
        if current_row < 0 or current_row >= len(self.current_assets):
            QMessageBox.information(self, "No Selection", "Please select an audio file to play.")
            return

        asset = self.current_assets[current_row]
        filepath = asset[2]

        if not os.path.exists(filepath):
            QMessageBox.warning(
                self, "File Not Found",
                f"The selected audio file could not be found:\n{filepath}",
            )
            return

        self.audio_player.setSource(QUrl.fromLocalFile(filepath))
        self.audio_player.play()
        self.status_label.setText(f"Playing: {asset[1]}")

    def stop_audio(self):
        self.audio_player.stop()
        self.status_label.setText("Playback stopped.")