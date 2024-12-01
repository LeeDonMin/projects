import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')  # Tkinter 백엔드 설정
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Polygon as MplPolygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
import json
plt.rc("font", family="Malgun Gothic")
class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Tkinter Matplotlib 이미지 편집 도구")
        self.root.geometry("1600x900")  # 기본 창 크기 설정

        self.image_paths = []
        self.current_index = -1
        self.images = {}  # 이미지를 저장할 딕셔너리로 변경
        self.labels = {}
        self.centers = {}
        self.polygons = {}
        self.polygon_labels = {}
        self.state_stacks = {}
        self.redo_stacks = {}
        self.check_vars = {}
        self.xy = []
        self.kmeans = None
        self.n_clusters = 0
        self.label_modify_mode = False  # 라벨 수정 모드 여부
        self.selected_regions = []      # 선택된 영역 저장

        # 폴리곤 수정 관련 변수
        self.editing_polygon = False
        self.selected_polygon_idx = None
        self.selected_vertex_idx = None
        self.dragging_vertex = False

        # 임시 폴리곤 목록 초기화
        self.temp_polygons = []
        self.temp_polygon_labels = []

        self.statebar()
        self.mainframe()

    def statebar(self):
        self.menu_bar = tk.Menu(self.root)
        self.settings_menu = tk.Menu(self.menu_bar, tearoff=0)

        self.root.config(menu=self.menu_bar)
        self.menu_bar.add_cascade(label="파일", menu=self.settings_menu)

        self.settings_menu.add_command(label="이미지 불러오기", command=self.load_images)
        self.settings_menu.add_command(label="마스크 저장하기", command=self.save_mask)
        self.settings_menu.add_command(label="테마 변경", command=self.change_theme)
        self.settings_menu.add_separator()
        self.settings_menu.add_command(label="설정 보기", command=self.show_settings)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="도움말", menu=self.help_menu)
        self.help_menu.add_command(label="앱 정보", command=self.about_app)

    def mainframe(self):
        self.sidebar_canvas = tk.Canvas(self.root, width=250)
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.Y)

        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.sidebar_canvas.yview)
        self.scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        self.sidebar_frame = ttk.Frame(self.sidebar_canvas)
        self.sidebar_canvas.create_window((0, 0), window=self.sidebar_frame, anchor="nw")

        self.sidebar_canvas.configure(yscrollcommand=self.scrollbar.set)

        def configure_sidebar(event):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

        self.sidebar_frame.bind("<Configure>", configure_sidebar)
        self.sidebar_canvas.bind_all("<MouseWheel>", self._on_mousewheel)  # 마우스 휠 스크롤 이벤트 연결

        self.image_list_label = ttk.Label(self.sidebar_frame, text="이미지 목록:")
        self.image_list_label.pack(pady=(5, 5), padx=10)
        self.image_listbox = tk.Listbox(self.sidebar_frame, height=5)
        self.image_listbox.pack(pady=5, padx=5, fill=tk.BOTH)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.toolbar_frame = ttk.Frame(self.plot_frame)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()
        self.status_label = ttk.Label(self.sidebar_frame, text="이미지를 불러오세요.")
        self.status_label.pack(pady=20)
        self.sidebarbutton()

    def _on_mousewheel(self, event):
        self.sidebar_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def sidebarbutton(self):
        # 채널 선택 변수 추가
        self.channel_vars = {
            'R': tk.BooleanVar(value=True),
            'G': tk.BooleanVar(value=True),
            'B': tk.BooleanVar(value=True)
        }
        # R, G, B 채널 체크박스 추가
        self.channel_frame = ttk.LabelFrame(self.sidebar_frame, text="채널 선택")
        self.channel_frame.pack(pady=10, padx=10, fill=tk.X)

        self.channel_r_checkbox = ttk.Checkbutton(
            self.channel_frame, text="R 채널", variable=self.channel_vars['R'], command=self.update_display_image)
        self.channel_r_checkbox.pack(anchor='w')

        self.channel_g_checkbox = ttk.Checkbutton(
            self.channel_frame, text="G 채널", variable=self.channel_vars['G'], command=self.update_display_image)
        self.channel_g_checkbox.pack(anchor='w')

        self.channel_b_checkbox = ttk.Checkbutton(
            self.channel_frame, text="B 채널", variable=self.channel_vars['B'], command=self.update_display_image)
        self.channel_b_checkbox.pack(anchor='w')

        # 투명도 조절 슬라이더 추가
        self.alpha_label = ttk.Label(self.sidebar_frame, text="투명도 조절:")
        self.alpha_label.pack(pady=(10, 5), padx=10)
        self.alpha_var = tk.DoubleVar(value=0.3)
        self.alpha_slider = ttk.Scale(self.sidebar_frame, from_=0.0, to=1.0, variable=self.alpha_var, orient='horizontal', command=self.update_display_image)
        self.alpha_slider.pack(pady=5, padx=10, fill=tk.X)

        # 필터 선택 변수 추가 (라디오 버튼으로 변경)
        self.filter_var = tk.StringVar(value="None")

        # 필터 선택 라디오 버튼 추가
        self.filter_frame = ttk.LabelFrame(self.sidebar_frame, text="필터 선택")
        self.filter_frame.pack(pady=10, padx=10, fill=tk.X)

        ttk.Radiobutton(self.filter_frame, text="None", variable=self.filter_var, value="None").pack(anchor='w')
        ttk.Radiobutton(self.filter_frame, text="Gaussian Blur", variable=self.filter_var, value="Gaussian Blur").pack(anchor='w')
        ttk.Radiobutton(self.filter_frame, text="Median Blur", variable=self.filter_var, value="Median Blur").pack(anchor='w')
        ttk.Radiobutton(self.filter_frame, text="Bilateral Filter", variable=self.filter_var, value="Bilateral Filter").pack(anchor='w')

        # 필터 크기 입력
        self.filter_size_label = ttk.Label(self.sidebar_frame, text="필터 크기 (홀수):")
        self.filter_size_label.pack(pady=(10, 5), padx=10)
        self.filter_size_entry = ttk.Entry(self.sidebar_frame)
        self.filter_size_entry.insert(0, "5")  # 기본값
        self.filter_size_entry.pack(pady=5, padx=10, fill=tk.X)

        self.select_cluster_button = ttk.Button(self.sidebar_frame, text="클러스터 선택", command=self.cluster)
        self.select_cluster_button.pack(pady=5, fill=tk.X)
        self.select_cluster_button.config(state=tk.DISABLED)

        # tk.Button으로 변경하여 배경색을 변경할 수 있게 함
        self.modify_labels_button = tk.Button(self.sidebar_frame, text="라벨 수정", command=self.modify_labels)
        self.modify_labels_button.pack(pady=5, fill=tk.X)
        self.modify_labels_button.config(state=tk.DISABLED)
        self.default_button_bg = self.modify_labels_button.cget('bg')  # 기본 배경색 저장

        self.generate_polygon_button = ttk.Button(self.sidebar_frame, text="폴리곤 생성", command=self.generate_polygons)
        self.generate_polygon_button.pack(pady=5, fill=tk.X)
        self.generate_polygon_button.config(state=tk.DISABLED)

        self.add_polygon_button = ttk.Button(self.sidebar_frame, text="폴리곤 추가", command=self.add_polygons)
        self.add_polygon_button.pack(pady=5, fill=tk.X)
        self.add_polygon_button.config(state=tk.DISABLED)

        self.edit_polygon_button = ttk.Button(self.sidebar_frame, text="폴리곤 수정", command=self.edit_polygons)
        self.edit_polygon_button.pack(pady=5, fill=tk.X)
        self.edit_polygon_button.config(state=tk.DISABLED)

        self.assign_label_button = ttk.Button(self.sidebar_frame, text="라벨 지정", command=self.assign_labels)
        self.assign_label_button.pack(pady=5, fill=tk.X)
        self.assign_label_button.config(state=tk.DISABLED)

        self.merge_polygons_button = ttk.Button(self.sidebar_frame, text="폴리곤 저장", command=self.merge_temp_polygons)
        self.merge_polygons_button.pack(pady=5, fill=tk.X)
        self.merge_polygons_button.config(state=tk.DISABLED)

        self.save_mask_button = ttk.Button(self.sidebar_frame, text="마스크 저장", command=self.save_mask)
        self.save_mask_button.pack(pady=5, fill=tk.X)
        self.save_mask_button.config(state=tk.DISABLED)

        # 폴리곤 생성 옵션 추가
        self.contour_mode_label = ttk.Label(self.sidebar_frame, text="컨투어 모드:")
        self.contour_mode_label.pack(pady=(20, 5), padx=10)
        self.contour_mode_var = tk.StringVar()
        self.contour_mode_var.set("RETR_EXTERNAL")  # 기본값
        self.contour_mode_menu = ttk.OptionMenu(
            self.sidebar_frame, self.contour_mode_var,
            "RETR_EXTERNAL",
            "RETR_EXTERNAL",
            "RETR_LIST",
            "RETR_CCOMP",
            "RETR_TREE"
        )
        self.contour_mode_menu.pack(pady=5, padx=10, fill=tk.X)

        self.approx_method_label = ttk.Label(self.sidebar_frame, text="근사화 방법:")
        self.approx_method_label.pack(pady=(20, 5), padx=10)
        self.approx_method_var = tk.StringVar()
        self.approx_method_var.set("CHAIN_APPROX_SIMPLE")  # 기본값
        self.approx_method_menu = ttk.OptionMenu(
            self.sidebar_frame, self.approx_method_var,
            "CHAIN_APPROX_SIMPLE",
            "CHAIN_APPROX_NONE",
            "CHAIN_APPROX_SIMPLE",
            "CHAIN_APPROX_TC89_L1",
            "CHAIN_APPROX_TC89_KCOS"
        )
        self.approx_method_menu.pack(pady=5, padx=10, fill=tk.X)

        self.min_area_label = ttk.Label(self.sidebar_frame, text="최소 면적:")
        self.min_area_label.pack(pady=(20, 5), padx=10)
        self.min_area_entry = ttk.Entry(self.sidebar_frame)
        self.min_area_entry.insert(0, "100")  # 기본값
        self.min_area_entry.pack(pady=5, padx=10, fill=tk.X)

        self.epsilon_label = ttk.Label(self.sidebar_frame, text="Epsilon 값 (0~1):")
        self.epsilon_label.pack(pady=(20, 5), padx=10)
        self.epsilon_entry = ttk.Entry(self.sidebar_frame)
        self.epsilon_entry.insert(0, "0.01")  # 기본값
        self.epsilon_entry.pack(pady=5, padx=10, fill=tk.X)

    def update_display_image(self, event=None):
        if self.current_index == -1:
            return

        if self.current_index not in self.images:
            return

        # 선택된 채널 가져오기
        selected_channels = [channel for channel, var in self.channel_vars.items() if var.get()]
        if not selected_channels:
            messagebox.showwarning("Warning", "하나 이상의 채널을 선택하세요.")
            return

        # 이미지 채널 분리
        image = self.images[self.current_index]
        channels = cv2.split(image)  # B, G, R 순서

        # 선택된 채널만 사용하여 이미지 구성
        channel_indices = {'B': 0, 'G': 1, 'R': 2}
        selected_indices = [channel_indices[ch] for ch in selected_channels]

        if len(selected_indices) == 1:
            # 단일 채널일 경우 그레이스케일로 표시
            display_image = channels[selected_indices[0]]
            self.ax.clear()
            self.ax.imshow(display_image, cmap='gray')
            self.ax.axis('off')
            self.canvas.draw()
        else:
            # 다중 채널일 경우 선택된 채널만 사용하여 컬러 이미지 구성
            new_channels = []
            for i in range(3):
                if i in selected_indices:
                    new_channels.append(channels[i])
                else:
                    new_channels.append(np.zeros_like(channels[0]))
            self.display_image = cv2.merge(new_channels)
            self.ax.clear()
            self.ax.imshow(self.display_image)
            self.ax.axis('off')
            self.canvas.draw()

    def cluster(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if self.current_index not in self.images:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        self.select_cluster_button.configure(style='Active.TButton')

        # 모든 이벤트 핸들러 해제 및 버튼 스타일 초기화
        self.disconnect_all_events()
        self.reset_button_styles()
        self.select_cluster_button.configure(style='Active.TButton')

        # 항상 원본 이미지로 표시
        self.ax.clear()
        self.ax.imshow(self.images[self.current_index])
        self.ax.axis('off')
        self.ax.set_title("원본 이미지 - 클러스터 포인트 선택")
        self.canvas.draw()

        # 클러스터 포인트 클릭 이벤트 연결
        self.cid_click_cluster = self.fig.canvas.mpl_connect("button_press_event", self.select_points)
        self.cid_key_press_cluster = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.xy = []  # 포인트 초기화
        self.canvas.draw()

    def select_points(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.xy.append([x, y])
        self.ax.plot(x, y, 'r+')
        self.canvas.draw()

    def key_press(self, event):
        if event.key == 'enter':
            if len(self.xy) >= 1:
                print('클러스터링 수행')
                self.fig.canvas.mpl_disconnect(self.cid_click_cluster)
                self.fig.canvas.mpl_disconnect(self.cid_key_press_cluster)
                self.cid_click_cluster = None
                self.cid_key_press_cluster = None
                self.clustering()

    def clustering(self):
        if self.current_index == -1:
            return

        if self.current_index not in self.images:
            return

        # 선택된 채널 가져오기
        selected_channels = [channel for channel, var in self.channel_vars.items() if var.get()]
        if not selected_channels:
            messagebox.showwarning("Warning", "하나 이상의 채널을 선택하세요.")
            return

        channel_indices = {'B': 0, 'G': 1, 'R': 2}
        selected_indices = [channel_indices[ch] for ch in selected_channels]

        # 이미지 데이터 가져오기
        image = self.images[self.current_index]

        # 필터 크기 가져오기
        try:
            filter_size = int(self.filter_size_entry.get())
            if filter_size % 2 == 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "필터 크기를 올바른 홀수로 입력하세요.")
            return

        # 필터 적용
        filtered_image = image.copy()

        filter_type = self.filter_var.get()
        if filter_type == "Gaussian Blur":
            filtered_image = cv2.GaussianBlur(filtered_image, (filter_size, filter_size), 0)
        elif filter_type == "Median Blur":
            filtered_image = cv2.medianBlur(filtered_image, filter_size)
        elif filter_type == "Bilateral Filter":
            filtered_image = cv2.bilateralFilter(filtered_image, filter_size, 75, 75)
        # 필터가 None인 경우 원본 이미지 사용

        # 클러스터링에 filtered_image 사용
        data = filtered_image.reshape((-1, 3))[:, selected_indices]

        # 클러스터 포인트의 색상 가져오기
        colors = [filtered_image[y, x, selected_indices] for x, y in self.xy]
        colors = np.array(colors)
        self.n_clusters = len(self.xy)

        # KMeans 클러스터링 수행
        self.kmeans = KMeans(n_clusters=self.n_clusters, init=colors, n_init=1, random_state=42)
        self.kmeans.fit(data)
        labels = self.kmeans.labels_.reshape(image.shape[:2])

        # 라벨과 색상 매핑
        centers = self.kmeans.cluster_centers_.astype(np.uint8)
        label_ids = np.unique(labels)
        self.centers[self.current_index] = {label: centers[idx] for idx, label in enumerate(label_ids)}

        self.labels[self.current_index] = labels

        # 세그먼트 이미지 생성
        segmented_img = np.zeros_like(image)
        for label, color in self.centers[self.current_index].items():
            # 선택된 채널에 따라 색상 구성
            full_color = np.zeros(3, dtype=np.uint8)
            for idx, ch_idx in enumerate(selected_indices):
                full_color[ch_idx] = color[idx]
            segmented_img[labels == label] = full_color

        self.segmented_img = segmented_img

        # 선택된 채널 정보 저장
        self.selected_channels = selected_channels
        self.selected_indices = selected_indices

        # 라벨 체크박스 프레임 생성
        if hasattr(self, 'label_frame'):
            self.label_frame.destroy()
        self.label_frame = ttk.Frame(self.sidebar_frame)
        self.label_frame.pack(fill=tk.BOTH, expand=True)

        self.check_vars = {}
        for label_id in np.unique(labels):
            self.add_label_checkbox(label_id)

        self.ax.clear()
        self.ax.imshow(segmented_img)
        self.ax.axis('off')
        self.ax.set_title("클러스터링 결과")
        self.canvas.draw()

        # 버튼 활성화
        self.generate_polygon_button.config(state=tk.NORMAL)
        self.add_polygon_button.config(state=tk.NORMAL)
        self.modify_labels_button.config(state=tk.NORMAL)
        self.edit_polygon_button.config(state=tk.NORMAL)
        self.assign_label_button.config(state=tk.NORMAL)
        self.merge_polygons_button.config(state=tk.NORMAL)
        self.save_mask_button.config(state=tk.NORMAL)

        # 상태 스택 초기화
        self.state_stacks[self.current_index] = []
        self.save_state()

    def add_label_checkbox(self, label_id):
        var = tk.BooleanVar(value=True)  # 기본으로 체크 상태
        self.check_vars[label_id] = var
        checkbutton = ttk.Checkbutton(self.label_frame, text=f"Label {label_id}", variable=var)
        checkbutton.pack(anchor='w')
        # 체크박스 상태 변경 시 자동으로 이미지 업데이트
        var.trace_add('write', self.on_checkbox_change)

    def on_checkbox_change(self, *args):
        self.show_selected()

    def show_selected(self):
        if self.current_index not in self.labels:
            return

        selected_items = [key for key, var in self.check_vars.items() if var.get()]

        # 선택된 채널 정보 가져오기
        selected_indices = self.selected_indices

        # 세그먼트 이미지와 동일한 크기의 배열 생성
        selected_pixels = np.zeros_like(self.segmented_img)

        for label in selected_items:
            label_mask = (self.labels[self.current_index] == label)
            # 라벨에 해당하는 색상 가져오기
            color = self.centers[self.current_index][label]
            # 전체 색상 배열 생성
            full_color = np.zeros(3, dtype=np.uint8)
            for idx, ch_idx in enumerate(selected_indices):
                full_color[ch_idx] = color[idx]
            selected_pixels[label_mask] = full_color

        self.ax.imshow(selected_pixels)
        # 투명도 적용
        alpha_value = self.alpha_var.get()
        self.ax.imshow(self.display_image, alpha=alpha_value)
        self.ax.axis('off')
        self.ax.set_title("선택된 라벨 보기")
        self.canvas.draw_idle()

    def modify_labels(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if self.current_index not in self.labels:
            messagebox.showwarning("Warning", "클러스터링을 먼저 수행하세요.")
            return

        if not self.label_modify_mode:
            # 라벨 수정 모드 시작
            self.label_modify_mode = True
            self.modify_labels_button.config(bg='green')
            # 이전에 존재하는 선택 도구를 해제
            if hasattr(self, 'selector') and self.selector is not None:
                self.selector.disconnect_events()
                del self.selector
                self.selector = None

            # RectangleSelector를 사용하여 영역 선택
            self.selector = RectangleSelector(self.ax, self.on_select_region, useblit=True,
                                              button=[1],  # 왼쪽 마우스 버튼
                                              minspanx=5, minspany=5,
                                              spancoords='pixels',
                                              interactive=False)
            self.canvas.draw_idle()
        else:
            # 라벨 수정 모드 종료
            self.label_modify_mode = False
            self.modify_labels_button.config(bg=self.default_button_bg)
            # 선택 도구 해제
            if hasattr(self, 'selector') and self.selector is not None:
                self.selector.disconnect_events()
                del self.selector
                self.selector = None

    def on_select_region(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # 이미지 경계 내로 좌표 보정
        x1, x2 = max(min(x1, x2), 0), min(max(x1, x2), self.labels[self.current_index].shape[1]-1)
        y1, y2 = max(min(y1, y2), 0), min(max(y1, y2), self.labels[self.current_index].shape[0]-1)

        # 선택된 영역 내에서 현재 표시 중인 라벨에 해당하는 픽셀 선택
        selected_items = [key for key, var in self.check_vars.items() if var.get()]

        region_mask = np.zeros_like(self.labels[self.current_index], dtype=bool)
        region_mask[y1:y2+1, x1:x2+1] = True

        label_mask = np.isin(self.labels[self.current_index], selected_items)

        combined_mask = region_mask & label_mask

        if np.any(combined_mask):
            # 상태 저장
            self.save_state()

            # 새로운 라벨 번호 입력 받기
            new_label_str = simpledialog.askstring("새 라벨 입력", "새로운 라벨 번호를 입력하세요:")
            if new_label_str is None:
                # 입력 취소 시
                return

            try:
                new_label = int(new_label_str)
            except ValueError:
                messagebox.showerror("Error", "올바른 정수를 입력하세요.")
                return

            # 라벨 업데이트
            self.labels[self.current_index][combined_mask] = new_label

            # 새로운 라벨의 색상 설정 (선택된 영역의 평균 색상)
            if new_label not in self.centers[self.current_index]:
                mean_color = np.mean(self.images[self.current_index][combined_mask], axis=0)
                self.centers[self.current_index][new_label] = mean_color.astype(np.uint8)
            else:
                # 기존 라벨일 경우 색상 업데이트
                mean_color = np.mean(self.images[self.current_index][self.labels[self.current_index] == new_label], axis=0)
                self.centers[self.current_index][new_label] = mean_color.astype(np.uint8)

            # 세그먼트 이미지 업데이트
            self.segmented_img[combined_mask] = self.centers[self.current_index][new_label]

            # 이미지 갱신
            self.show_selected()

            # 체크박스 리스트에 라벨 추가
            if new_label not in self.check_vars:
                self.add_label_checkbox(new_label)

        # 사각형 제거
        self.selector.set_visible(False)
        self.canvas.draw_idle()

    def generate_polygons(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if self.current_index not in self.labels:
            messagebox.showwarning("Warning", "클러스터링을 먼저 수행하세요.")
            return

        # 임시 폴리곤 초기화
        self.temp_polygons = []
        self.temp_polygon_labels = []

        # 선택된 라벨 기반으로 이미지 생성
        selected_items = [key for key, var in self.check_vars.items() if var.get()]
        if not selected_items:
            messagebox.showwarning("Warning", "하나 이상의 라벨을 선택하세요.")
            return

        # 사용자 입력 값 가져오기
        try:
            min_area = int(self.min_area_entry.get())
            epsilon_value = float(self.epsilon_entry.get())
        except ValueError:
            messagebox.showerror("Error", "최소 면적과 Epsilon 값을 올바르게 입력하세요.")
            return

        contour_mode_str = self.contour_mode_var.get()
        approx_method_str = self.approx_method_var.get()

        # 컨투어 모드 매핑
        contour_modes = {
            "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
            "RETR_LIST": cv2.RETR_LIST,
            "RETR_CCOMP": cv2.RETR_CCOMP,
            "RETR_TREE": cv2.RETR_TREE
        }
        contour_mode = contour_modes.get(contour_mode_str, cv2.RETR_EXTERNAL)

        # 근사화 방법 매핑
        approx_methods = {
            "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
            "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
            "CHAIN_APPROX_TC89_L1": cv2.CHAIN_APPROX_TC89_L1,
            "CHAIN_APPROX_TC89_KCOS": cv2.CHAIN_APPROX_TC89_KCOS
        }
        approx_method = approx_methods.get(approx_method_str, cv2.CHAIN_APPROX_SIMPLE)

        # 선택된 라벨의 마스크 생성
        mask = np.isin(self.labels[self.current_index], selected_items).astype(np.uint8) * 255

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, contour_mode, approx_method)

        # 폴리곤 생성
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                epsilon = epsilon_value * cv2.arcLength(cnt, True)
                polygon = cv2.approxPolyDP(cnt, epsilon, True)
                if len(polygon) >= 3:
                    self.temp_polygons.append(polygon.reshape(-1, 2))
                    self.temp_polygon_labels.append('unknown')  # 초기 라벨은 'unknown'

        # 폴리곤 그리기
        self.update_polygon_image()

        # 버튼 스타일 및 상태 업데이트
        self.reset_button_styles()
        self.edit_polygon_button.configure(state=tk.NORMAL)
        self.modify_labels_button.configure(state=tk.NORMAL)
        self.assign_label_button.configure(state=tk.NORMAL)
        self.merge_polygons_button.configure(state=tk.NORMAL)

        messagebox.showinfo("폴리곤 생성 완료", "폴리곤이 생성되었습니다. '라벨 지정' 버튼을 눌러 라벨을 지정하세요.")

        # 상태 저장
        self.save_state()

    def add_polygons(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if self.current_index not in self.labels:
            messagebox.showwarning("Warning", "클러스터링을 먼저 수행하세요.")
            return

        # 기존 임시 폴리곤을 유지하고 추가로 폴리곤 생성
        # 사용자 입력 값 가져오기
        try:
            min_area = int(self.min_area_entry.get())
            epsilon_value = float(self.epsilon_entry.get())
        except ValueError:
            messagebox.showerror("Error", "최소 면적과 Epsilon 값을 올바르게 입력하세요.")
            return

        contour_mode_str = self.contour_mode_var.get()
        approx_method_str = self.approx_method_var.get()

        # 컨투어 모드 매핑
        contour_modes = {
            "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
            "RETR_LIST": cv2.RETR_LIST,
            "RETR_CCOMP": cv2.RETR_CCOMP,
            "RETR_TREE": cv2.RETR_TREE
        }
        contour_mode = contour_modes.get(contour_mode_str, cv2.RETR_EXTERNAL)

        # 근사화 방법 매핑
        approx_methods = {
            "CHAIN_APPROX_NONE": cv2.CHAIN_APPROX_NONE,
            "CHAIN_APPROX_SIMPLE": cv2.CHAIN_APPROX_SIMPLE,
            "CHAIN_APPROX_TC89_L1": cv2.CHAIN_APPROX_TC89_L1,
            "CHAIN_APPROX_TC89_KCOS": cv2.CHAIN_APPROX_TC89_KCOS
        }
        approx_method = approx_methods.get(approx_method_str, cv2.CHAIN_APPROX_SIMPLE)

        # 선택된 라벨의 마스크 생성
        selected_items = [key for key, var in self.check_vars.items() if var.get()]
        if not selected_items:
            messagebox.showwarning("Warning", "하나 이상의 라벨을 선택하세요.")
            return

        mask = np.isin(self.labels[self.current_index], selected_items).astype(np.uint8) * 255

        # 컨투어 찾기
        contours, _ = cv2.findContours(mask, contour_mode, approx_method)

        # 폴리곤 생성
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                epsilon = epsilon_value * cv2.arcLength(cnt, True)
                polygon = cv2.approxPolyDP(cnt, epsilon, True)
                if len(polygon) >= 3:
                    self.temp_polygons.append(polygon.reshape(-1, 2))
                    self.temp_polygon_labels.append('unknown')  # 초기 라벨은 'unknown'

        # 폴리곤 그리기
        self.update_polygon_image()

        messagebox.showinfo("폴리곤 추가 생성 완료", "폴리곤이 추가로 생성되었습니다. '라벨 지정' 버튼을 눌러 라벨을 지정하세요.")

        # 상태 저장
        self.save_state()

    def update_polygon_image(self, show_vertices=False):
        # 현재 축 범위 저장
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # 폴리곤을 그린 이미지를 업데이트합니다.
        self.ax.clear()
        self.ax.imshow(self.images[self.current_index])
        self.ax.axis('off')

        # 기존 폴리곤 그리기
        if self.current_index in self.polygons:
            for idx, polygon in enumerate(self.polygons[self.current_index]):
                label = self.polygon_labels[self.current_index][idx]
                color = self.get_color_by_label(label)
                mpl_polygon = MplPolygon(polygon, closed=True, edgecolor=color, fill=False, linewidth=2)
                self.ax.add_patch(mpl_polygon)

                if show_vertices:
                    # 버텍스 표시
                    self.ax.scatter(polygon[:, 0], polygon[:, 1], color='red', s=20)

        # 임시 폴리곤 그리기
        for idx, polygon in enumerate(self.temp_polygons):
            label = self.temp_polygon_labels[idx]
            color = self.get_color_by_label(label)
            mpl_polygon = MplPolygon(polygon, closed=True, edgecolor=color, linestyle='--', fill=False, linewidth=2)
            self.ax.add_patch(mpl_polygon)

            if show_vertices:
                # 버텍스 표시
                self.ax.scatter(polygon[:, 0], polygon[:, 1], color='orange', s=20)

        # 축 범위 복원
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw()

    def get_color_by_label(self, label):
        """
        라벨에 따라 색상을 반환합니다.
        같은 라벨은 같은 색상을 가집니다.
        """
        label_colors = {
            'background': 'blue',
            'film': 'green',
            'fold': 'yellow',
            'overlap': 'red',
            'unknown': 'gray'
        }
        return label_colors.get(label, 'gray')

    def edit_polygons(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if not self.polygons.get(self.current_index) and not self.temp_polygons:
            messagebox.showwarning("Warning", "폴리곤이 없습니다.")
            return

        # 모든 이벤트 핸들러 해제 및 버튼 스타일 초기화
        self.disconnect_all_events()
        self.reset_button_styles()
        self.edit_polygon_button.configure(style='Active.TButton')

        self.editing_polygon = True
        self.status_label.config(text="폴리곤 수정 모드입니다. 버텍스를 선택하여 이동하거나, 우클릭하여 버텍스를 추가/삭제하세요.")

        # 폴리곤을 그린 이미지 표시
        self.update_polygon_image(show_vertices=True)

        # 이벤트 연결
        self.cid_click_polygon = self.fig.canvas.mpl_connect("button_press_event", self.on_click_vertex)
        self.cid_motion_polygon = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion_vertex)
        self.cid_release_polygon = self.fig.canvas.mpl_connect("button_release_event", self.on_release_vertex)

    def on_click_vertex(self, event):
        if not self.editing_polygon:
            return

        if event.x is None or event.y is None:
            return

        # 좌표 변환
        x, y = self.ax.transData.inverted().transform((event.x, event.y))

        if event.button == 1:  # Left click
            # 버텍스 선택
            # 먼저 임시 폴리곤에서 검색
            for idx_p, polygon in enumerate(self.temp_polygons):
                for idx_v, vertex in enumerate(polygon):
                    vx, vy = vertex
                    distance = np.hypot(vx - x, vy - y)
                    if distance < 5:  # 버텍스와의 거리 임계값
                        self.selected_polygon_type = 'temp'
                        self.selected_polygon_idx = idx_p
                        self.selected_vertex_idx = idx_v
                        self.dragging_vertex = True
                        return
            # 기존 폴리곤에서 검색
            if self.current_index in self.polygons:
                for idx_p, polygon in enumerate(self.polygons[self.current_index]):
                    for idx_v, vertex in enumerate(polygon):
                        vx, vy = vertex
                        distance = np.hypot(vx - x, vy - y)
                        if distance < 5:
                            self.selected_polygon_type = 'existing'
                            self.selected_polygon_idx = idx_p
                            self.selected_vertex_idx = idx_v
                            self.dragging_vertex = True
                            return
        elif event.button == 3:  # Right click
            # 버텍스 추가 또는 삭제
            # 임시 폴리곤부터 처리
            for idx_p, polygon in enumerate(self.temp_polygons):
                polygon_np = np.array(polygon)
                # 버텍스 삭제
                for idx_v, vertex in enumerate(polygon_np):
                    vx, vy = vertex
                    distance = np.hypot(vx - x, vy - y)
                    if distance < 5:
                        self.temp_polygons[idx_p] = np.delete(polygon_np, idx_v, axis=0)
                        self.update_polygon_image(show_vertices=True)
                        return
                # 버텍스 추가 (선분 위인지 확인)
                num_vertices = len(polygon_np)
                for idx_v in range(num_vertices):
                    x1, y1 = polygon_np[idx_v]
                    x2, y2 = polygon_np[(idx_v + 1) % num_vertices]
                    if self.is_point_near_line(x, y, x1, y1, x2, y2):
                        new_vertex = np.array([[int(x), int(y)]])
                        self.temp_polygons[idx_p] = np.insert(polygon_np, idx_v + 1, new_vertex, axis=0)
                        self.update_polygon_image(show_vertices=True)
                        return
            # 기존 폴리곤 처리
            if self.current_index in self.polygons:
                for idx_p, polygon in enumerate(self.polygons[self.current_index]):
                    polygon_np = np.array(polygon)
                    # 버텍스 삭제
                    for idx_v, vertex in enumerate(polygon_np):
                        vx, vy = vertex
                        distance = np.hypot(vx - x, vy - y)
                        if distance < 5:
                            self.polygons[self.current_index][idx_p] = np.delete(polygon_np, idx_v, axis=0)
                            self.update_polygon_image(show_vertices=True)
                            return
                    # 버텍스 추가 (선분 위인지 확인)
                    num_vertices = len(polygon_np)
                    for idx_v in range(num_vertices):
                        x1, y1 = polygon_np[idx_v]
                        x2, y2 = polygon_np[(idx_v + 1) % num_vertices]
                        if self.is_point_near_line(x, y, x1, y1, x2, y2):
                            new_vertex = np.array([[int(x), int(y)]])
                            self.polygons[self.current_index][idx_p] = np.insert(polygon_np, idx_v + 1, new_vertex, axis=0)
                            self.update_polygon_image(show_vertices=True)
                            return

    def is_point_near_line(self, px, py, x1, y1, x2, y2, threshold=5):
        # 선분 (x1, y1)-(x2, y2)와 점 (px, py) 사이의 거리를 계산
        line_mag = np.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-6:
            return False
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        if u < 0 or u > 1:
            return False
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        distance = np.hypot(px - ix, py - iy)
        return distance < threshold

    def on_motion_vertex(self, event):
        if not self.dragging_vertex:
            return

        if event.x is None or event.y is None:
            return

        # 좌표 변환
        x, y = self.ax.transData.inverted().transform((event.x, event.y))

        # 버텍스 위치 업데이트
        if self.selected_polygon_type == 'temp':
            self.temp_polygons[self.selected_polygon_idx][self.selected_vertex_idx] = [int(x), int(y)]
        elif self.selected_polygon_type == 'existing':
            self.polygons[self.current_index][self.selected_polygon_idx][self.selected_vertex_idx] = [int(x), int(y)]

        # 폴리곤 및 버텍스 업데이트
        self.update_polygon_image(show_vertices=True)

    def on_release_vertex(self, event):
        if self.dragging_vertex:
            self.dragging_vertex = False
            self.selected_polygon_idx = None
            self.selected_vertex_idx = None
            self.selected_polygon_type = None
            self.status_label.config(text="폴리곤 수정 완료.")
            # 상태 저장
            self.save_state()

    def assign_labels(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "이미지를 먼저 불러오세요.")
            return

        if not self.polygons.get(self.current_index) and not self.temp_polygons:
            messagebox.showwarning("Warning", "폴리곤이 없습니다.")
            return

        # 모든 이벤트 핸들러 해제 및 버튼 스타일 초기화
        self.disconnect_all_events()
        self.reset_button_styles()
        self.assign_label_button.configure(style='Active.TButton')

        self.assigning_label = True
        self.status_label.config(text="라벨 지정 모드입니다. 폴리곤을 클릭하여 라벨을 지정하세요.")

        # 폴리곤을 그린 이미지 표시
        self.update_polygon_image()

        # 폴리곤 라벨링 클릭 이벤트 연결
        self.cid_click_label = self.fig.canvas.mpl_connect("button_press_event", self.on_click_label)

    def on_click_label(self, event):
        if not self.assigning_label:
            return  # 라벨 지정 모드가 아닐 때는 무시

        if event.x is None or event.y is None:
            return

        # 좌표 변환
        x, y = self.ax.transData.inverted().transform((event.x, event.y))

        clicked_point = (x, y)

        # 임시 폴리곤부터 검사
        for idx, polygon in enumerate(self.temp_polygons):
            # OpenCV의 pointPolygonTest 사용
            result = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (int(x), int(y)), False)
            if result >= 0:
                # 폴리곤 내부 클릭
                current_label = self.temp_polygon_labels[idx]
                label = simpledialog.askstring("폴리곤 라벨 지정", "이 폴리곤의 라벨을 입력하세요 (예: background, film, fold, overlap):", initialvalue=current_label)
                if label:
                    self.temp_polygon_labels[idx] = label
                    # 색상 변경 (라벨에 따라 색상 지정)
                    self.update_polygon_image()
                return  # 하나의 폴리곤에만 라벨을 지정하도록 함

        # 기존 폴리곤 검사
        if self.current_index in self.polygons:
            for idx, polygon in enumerate(self.polygons[self.current_index]):
                # OpenCV의 pointPolygonTest 사용
                result = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (int(x), int(y)), False)
                if result >= 0:
                    # 폴리곤 내부 클릭
                    current_label = self.polygon_labels[self.current_index][idx]
                    label = simpledialog.askstring("폴리곤 라벨 지정", "이 폴리곤의 라벨을 입력하세요 (예: background, film, fold, overlap):", initialvalue=current_label)
                    if label:
                        self.polygon_labels[self.current_index][idx] = label
                        # 색상 변경 (라벨에 따라 색상 지정)
                        self.update_polygon_image()
                    return  # 하나의 폴리곤에만 라벨을 지정하도록 함

    def merge_temp_polygons(self):
        if not self.temp_polygons:
            messagebox.showwarning("Warning", "임시 폴리곤이 없습니다.")
            return

        # 폴리곤 딕셔너리가 없을 경우 초기화
        if self.current_index not in self.polygons:
            self.polygons[self.current_index] = []
            self.polygon_labels[self.current_index] = []

        # 임시 폴리곤을 기존 폴리곤에 합침
        self.polygons[self.current_index].extend(self.temp_polygons)
        self.polygon_labels[self.current_index].extend(self.temp_polygon_labels)

        # 임시 폴리곤 초기화
        self.temp_polygons = []
        self.temp_polygon_labels = []

        # 폴리곤 업데이트
        self.update_polygon_image()

        messagebox.showinfo("폴리곤 저장 완료", "임시 폴리곤이 저장되었습니다.")

        # 상태 저장
        self.save_state()

    def disconnect_all_events(self):
        # 모든 이벤트 핸들러 해제
        if hasattr(self, 'cid_click_polygon') and self.cid_click_polygon:
            self.fig.canvas.mpl_disconnect(self.cid_click_polygon)
            self.cid_click_polygon = None
        if hasattr(self, 'cid_motion_polygon') and self.cid_motion_polygon:
            self.fig.canvas.mpl_disconnect(self.cid_motion_polygon)
            self.cid_motion_polygon = None
        if hasattr(self, 'cid_release_polygon') and self.cid_release_polygon:
            self.fig.canvas.mpl_disconnect(self.cid_release_polygon)
            self.cid_release_polygon = None
        if hasattr(self, 'cid_click_label') and self.cid_click_label:
            self.fig.canvas.mpl_disconnect(self.cid_click_label)
            self.cid_click_label = None
        if hasattr(self, 'selector') and self.selector is not None:
            self.selector.disconnect_events()
            del self.selector
            self.selector = None

    def reset_button_styles(self):
        # 모든 버튼의 스타일을 기본값으로 재설정
        self.select_cluster_button.configure(style='TButton')
        self.modify_labels_button.config(bg=self.default_button_bg)
        self.generate_polygon_button.configure(style='TButton')
        self.add_polygon_button.configure(style='TButton')
        self.edit_polygon_button.configure(style='TButton')
        self.assign_label_button.configure(style='TButton')
        self.merge_polygons_button.configure(style='TButton')

    def on_image_select(self, event):
        if not self.image_listbox.curselection():
            return
        index = self.image_listbox.curselection()[0]
        if index == self.current_index:
            return  # 이미 선택된 이미지

        self.current_index = index
        image_path = self.image_paths[self.current_index]

        # 이미지를 로드하지 않았다면 로드
        if self.current_index not in self.images:
            # OpenCV로 이미지 로드 및 RGB로 변환
            image = cv2.imread(image_path)
            if image is None:
                messagebox.showerror("Error", f"이미지를 불러올 수 없습니다: {image_path}")
                return
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images[self.current_index] = image

        # 버튼 상태 활성화
        self.select_cluster_button.config(state=tk.NORMAL)
        self.modify_labels_button.config(state=tk.NORMAL)
        self.generate_polygon_button.config(state=tk.NORMAL)
        self.add_polygon_button.config(state=tk.NORMAL)
        self.edit_polygon_button.config(state=tk.NORMAL)
        self.assign_label_button.config(state=tk.NORMAL)
        self.merge_polygons_button.config(state=tk.NORMAL)
        self.save_mask_button.config(state=tk.NORMAL)

        # 상태 스택 초기화
        if self.current_index not in self.state_stacks:
            self.state_stacks[self.current_index] = []
            self.redo_stacks[self.current_index] = []

        self.display_current_image()
        self.update_display_image()

    def display_current_image(self):
        if self.current_index == -1:
            return
        else:
            if self.current_index not in self.images:
                return
            self.ax.clear()
            self.ax.imshow(self.images[self.current_index])
            self.ax.axis('off')
            self.canvas.draw()

    def load_images(self):
        # 파일 다이얼로그 열기 (여러 파일 선택 가능)
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_paths:
            return

        for file_path in file_paths:
            # 중복 체크
            if file_path in self.image_paths:
                continue

            self.image_paths.append(file_path)

            # 이미지 리스트에 추가
            base_name = os.path.basename(file_path)
            self.image_listbox.insert(tk.END, base_name)

        if self.image_paths:
            # 자동 선택 첫 이미지
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(0)
            self.on_image_select(None)

    def save_mask(self):
        if self.current_index == -1:
            messagebox.showwarning("Warning", "저장할 이미지가 없습니다.")
            return

        if not self.polygons.get(self.current_index):
            messagebox.showwarning("Warning", "폴리곤이 없습니다.")
            return

        height, width, _ = self.images[self.current_index].shape
        mask = np.zeros((height, width, 3), dtype=np.uint8)

        # 클래스별로 코드 매핑
        class_codes = {
            'film': [1, 0, 0],
            'fold': [1, 1, 0],
            'overlap': [1, 0, 1]
        }

        for polygon, label in zip(self.polygons[self.current_index], self.polygon_labels[self.current_index]):
            code = class_codes.get(label, [0, 0, 0])  # 기본값은 [0, 0, 0]
            # 폴리곤 영역에 코드 채우기
            polygon_mask = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [polygon.astype(np.int32)], code)

            # 기존 마스크와 새로운 폴리곤 마스크를 결합 (OR 연산)
            mask = np.maximum(mask, polygon_mask)

        mask_scaled = (mask * 255).astype(np.uint8)

        # 저장 경로 설정
        base_name = os.path.splitext(os.path.basename(self.image_paths[self.current_index]))[0]
        save_path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=f"{base_name}_mask.png", filetypes=[("PNG files", "*.png")])

        if save_path:
            # 마스크를 PNG 파일로 저장
            cv2.imwrite(save_path, cv2.cvtColor(mask_scaled, cv2.COLOR_RGB2BGR))
            messagebox.showinfo("Success", f"마스크가 PNG 파일로 저장되었습니다: {save_path}")

    def save_state(self):
        # 현재 상태를 스택에 저장
        state = {
            'polygons': [polygon.copy() for polygon in self.polygons.get(self.current_index, [])],
            'polygon_labels': self.polygon_labels.get(self.current_index, [])[:]
        }
        self.state_stacks[self.current_index].append(state)
        self.redo_stacks[self.current_index] = []  # 새로운 작업이 추가되면 redo 스택 초기화

        # 상태 스택 크기 제한 (예: 최대 10개)
        max_stack_size = 10
        if len(self.state_stacks[self.current_index]) > max_stack_size:
            self.state_stacks[self.current_index].pop(0)

    def undo(self, event=None):
        if self.current_index in self.state_stacks and self.state_stacks[self.current_index]:
            state = self.state_stacks[self.current_index].pop()
            self.redo_stacks[self.current_index].append(state)
            self.restore_state()
        else:
            messagebox.showinfo("Undo", "더 이상 되돌릴 작업이 없습니다.")

    def redo(self, event=None):
        if self.current_index in self.redo_stacks and self.redo_stacks[self.current_index]:
            state = self.redo_stacks[self.current_index].pop()
            self.state_stacks[self.current_index].append(state)
            self.restore_state()
        else:
            messagebox.showinfo("Redo", "더 이상 다시 실행할 작업이 없습니다.")

    def restore_state(self):
        if self.state_stacks[self.current_index]:
            state = self.state_stacks[self.current_index][-1]
            self.polygons[self.current_index] = [polygon.copy() for polygon in state['polygons']]
            self.polygon_labels[self.current_index] = state['polygon_labels'][:]
            self.update_polygon_image()
        else:
            # 초기 상태로 복원
            self.polygons[self.current_index] = []
            self.polygon_labels[self.current_index] = []
            self.update_display_image()

    def change_theme(self):
        messagebox.showinfo("Theme", "Theme Change Window")

    def show_settings(self):
        messagebox.showinfo("Settings", "Settings Window")

    def about_app(self):
        messagebox.showinfo("About", "This is a Tkinter application example.")

    def run(self):
        # 키 이벤트 바인딩
        self.root.bind_all("<Control-z>", self.undo)
        self.root.bind_all("<Control-Z>", self.undo)
        self.root.bind_all("<Control-Shift-Z>", self.redo)
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    editor = ImageEditor(root)
    editor.run()
