from __future__ import annotations
import sys
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt, Slot, QThread, QSize
from PySide6.QtGui import QGuiApplication, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow,  QGraphicsScene,
     QFileDialog, QMessageBox,  QProgressDialog
)
from shp_converter_mainwindow import Ui_MainWindow
from dbfread import DBF
import os

from shp_converter import ConversionWorker
from mixed_converter import MixedConverter
from sewerage_converter import SewerageConverter
from concurrent.futures import ThreadPoolExecutor
from shp_union_export_worker import ExportWorker

class WideProgressDialog(QProgressDialog):
    def __init__(self, *args, min_width=600, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_width = min_width

    def sizeHint(self) -> QSize:
        sh = super().sizeHint()
        return QSize(max(sh.width(), self._min_width), sh.height())

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # worker 스레드 관련 변수 선언
        self.worker_thread = None
        self.worker = None

        # 백그라운드 로드를 위한 스레드 풀 (IO/계산 작업용)
        self._executor = ThreadPoolExecutor(max_workers=1)

        # 그래픽스 뷰 위젯에 이미지 파일 로딩 
        scene = QGraphicsScene(self)
        pixmap = QtGui.QPixmap("C:\Dev_Python\Workspace\shapeconverter\shp_convert_image.png")  # 이미지
        #pixmap = QtGui.QPixmap("C:\Workspace\shapeconvert\shp_convert_image.png")  # 이미지
        if not pixmap.isNull():
            scene.addPixmap(pixmap)
        else:
            print("이미지 파일을 로드할 수 없습니다.")
        self.ui.graphicsView.setScene(scene)
    
        self.is_working = False # 작업 중 상태 플래그
        self.was_canceled = None

        # GUI 초기화
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

        # 탭 변경 이벤트 연결
        self.ui.tabWidget.currentChanged.connect(self.on_tab_changed)

        # 닫기 버튼 클릭 이벤트 연결
        self.ui.btnClose.clicked.connect(self.on_window_close_clicked)

        # tab_1 이벤트 처리
        self.ui.btnBrowse.clicked.connect(self.on_browse_clicked)
        self.ui.btnSaveFolderBrowse.clicked.connect(self.on_save_folder_browse_clicked)
        self.ui.btnApply.clicked.connect(self.on_apply_clicked)
        self.ui.btnStart.clicked.connect(self.start_avrg_conversion)
        self.ui.btnStop.clicked.connect(self.cancel_avrg_conversion)
        self.ui.cmbResamplingIntervals.currentTextChanged.connect(self.on_resampling_intervals_changed)

        # tab_2 이벤트 처리
        self.ui.btnBrowse_2.clicked.connect(self.on_browse_clicked)
        self.ui.btnBrowsePointshp.clicked.connect(self.on_browse_clicked)
        self.ui.btnSaveFolderBrowse_2.clicked.connect(self.on_save_folder_browse_clicked)
        self.ui.btnApply_2.clicked.connect(self.on_apply_mix_clicked)
        self.ui.btnStart_2.clicked.connect(self.start_mixed_conversion)
        self.ui.btnStop_2.clicked.connect(self.cancel_mixed_conversion)
        self.ui.btnGDFExport.clicked.connect(self.export_gdf)

        # tab_3 이벤트 처리
        self.ui.btnBrowse_3.clicked.connect(self.on_browse_clicked)
        self.ui.btnSaveFolderBrowse_3.clicked.connect(self.on_save_folder_browse_clicked)
        self.ui.btnApply_3.clicked.connect(self.on_apply_sewerage_clicked)
        self.ui.btnStart_3.clicked.connect(self.start_sewerage_conversion)
        self.ui.btnStop_3.clicked.connect(self.cancel_sewerage_conversion)

        # tab_4 이벤트 처리
        self.ui.btWorkFolderBrowse.clicked.connect(self.on_work_folder_browse_clicked)
        self.ui.btSourceFolderBrowse.clicked.connect(self.on_source_folder_browse_clicked)
        self.ui.btGdfListOpen.clicked.connect(self.on_Gdf_List_Open_clicked)
        self.ui.btAllFilesAdd.clicked.connect(self.on_add_all_files_clicked)
        self.ui.btSelectedFilesAdd.clicked.connect(self.on_add_selected_files_clicked)
        self.ui.btRemoveSelected.clicked.connect(self.on_remove_selected_files_clicked)
        self.ui.btRemoveAll.clicked.connect(self.on_remove_all_clicked)
        self.ui.btAnalysisWorkStart.clicked.connect(self.start_batch_analysis)
        self.ui.btAnalysisWorkStop.clicked.connect(self.stop_batch_analysis)



    def setup_tab1(self):
        # tab1 UI 초기화
        self.ui.cmbFacilType.clear()
        self.ui.cmbFacilType.addItems(["상수관로", "가스관로", "난방관로", "전기관로", "통신관로"])
        self.ui.cmbFacilType.setPlaceholderText("관로유형 선택")
        self.ui.ledShapeFile.clear()
        self.ui.ledShapeFile.setPlaceholderText("변환할 shape 파일을 선택하여 주세요")
        self.ui.ledSaveFolder.clear()
        self.ui.ledSaveFolder.setPlaceholderText("저장 폴더를 선택하여 주세요")
        self.ui.cmbFeatureID.clear()
        self.ui.cmbFeatureID.setPlaceholderText("관로ID 필드 선택")
        self.ui.cmbDepthSelect.clear()
        self.ui.cmbDepthSelect.setPlaceholderText("평균심도 필드 선택")
        self.ui.cmbSizeSelect.clear()
        self.ui.cmbSizeSelect.setPlaceholderText("구경 크기 필드 선택")
        self.ui.cmbCircleSeg.clear()
        self.ui.cmbCircleSeg.addItems(["16", "24", "32", "64", "128", "256"])
        self.ui.cmbCircleSeg.setPlaceholderText("원형 단면 분할 갯수를 선택하여 주세요")  # 기본값 설정
        self.ui.cmbResamplingIntervals.clear()
        self.ui.cmbResamplingIntervals.addItems(["0.0","1.0", "2.0", "3.0", "5.0", "10.0", "20.0"])
        self.ui.cmbResamplingIntervals.setPlaceholderText("선형 중간점 간격을 설정하여 주세요")
        self.ui.ledConnectTolerance.clear()
        self.ui.ledConnectTolerance.setPlaceholderText("끝점 연결 허용오차를 입력하여 주세요")
        self.ui.cmbExportFileFormat.clear()
        self.ui.cmbExportFileFormat.addItems(["OBJ", "GLB", "GLTF"])
        self.ui.cmbExportFileFormat.setPlaceholderText("내보낼 파일 형식을 선택하여 주세요")
        self.ui.chkResampleOption_1.setChecked(False)
        self.ui.btnStart.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop.setEnabled(False)  # 변환중지 버튼 비활성화
        self.ui.lblProgressMsg.clear()
        self.ui.lblProgressMsg.setText("변환작업 대기 중 ...")
        self.ui.progressBar.setValue(0)
        self.ui.progressBar.setRange(0, 100)  # 프로그래스바 범위 설정

    def setup_tab2(self):
        # tab2 UI 초기화
        self.ui.cmbFacilType_2.clear()
        self.ui.cmbFacilType_2.addItems(["상수관로", "가스관로", "난방관로", "전기관로", "통신관로"])
        self.ui.ledShapeFile_2.clear()
        self.ui.ledShapeFile_2.setPlaceholderText("변환할 관로 shape 파일을 선택하여 주세요")
        self.ui.ledPointShapeFile.clear()
        self.ui.ledPointShapeFile.setPlaceholderText("변환할 심도 shape 파일을 선택하여 주세요")
        self.ui.ledSaveFolder_2.clear()
        self.ui.ledSaveFolder_2.setPlaceholderText("저장 폴더를 선택하여 주세요")
        self.ui.cmbFeatureID_2.clear()
        self.ui.cmbFeatureID_2.setPlaceholderText("관로ID 필드 선택")        
        self.ui.cmbAvrgDepthSelect.clear()
        self.ui.cmbAvrgDepthSelect.setPlaceholderText("평균심도 필드 선택")
        self.ui.cmbSizeSelect_2.clear()
        self.ui.cmbSizeSelect_2.setPlaceholderText("구경 크기 필드 선택")
        self.ui.cmbPointDepthSelect.clear()
        self.ui.cmbPointDepthSelect.setPlaceholderText("포인트 심도 필드 선택")
        self.ui.cmbCircleSeg_2.clear()
        self.ui.cmbCircleSeg_2.addItems(["16", "24", "32", "64", "128", "256"])
        self.ui.cmbCircleSeg_2.setPlaceholderText("원형 단면 분할 갯수를 선택하여 주세요")  # 기본값 설정
        self.ui.ledConnectTolerance_2.clear()
        self.ui.ledConnectTolerance_2.setPlaceholderText("끝점 연결 허용오차를 입력하여 주세요")
        self.ui.cmbExportFileFormat_2.clear()
        self.ui.cmbExportFileFormat_2.addItems(["OBJ", "GLB", "GLTF"])
        self.ui.cmbExportFileFormat_2.setPlaceholderText("내보낼 파일 형식을 선택하여 주세요")
        self.ui.btnStart_2.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop_2.setEnabled(False)  # 변환중지 버튼 비활성화
        self.ui.lblProgressMsg_2.clear()
        self.ui.lblProgressMsg_2.setText("변환작업 대기 중 ...")
        self.ui.progressBar_2.setValue(0)
        self.ui.progressBar_2.setRange(0, 100)  # 프로그래스바 범위 설정

    def setup_tab3(self):
        # tab3 UI 초기화
        self.ui.cmbFacilType_3.clear()
        self.ui.cmbFacilType_3.addItems(["하수관로"])        
        self.ui.cmbFacilType_3.setCurrentText("하수관로")  # 기본값 설정
        self.ui.ledShapeFile_3.clear()
        self.ui.ledShapeFile_3.setPlaceholderText("변환할 shape 파일을 선택하여 주세요")
        self.ui.ledSaveFolder_3.clear()
        self.ui.ledSaveFolder_3.setPlaceholderText("저장 폴더를 선택하여 주세요")
        self.ui.cmbStartDepthSelect.clear()
        self.ui.cmbStartDepthSelect.setPlaceholderText("시작심도 필드 선택")
        self.ui.cmbEndDepthSelect.clear()
        self.ui.cmbEndDepthSelect.setPlaceholderText("종료심도 필드 선택")
        self.ui.cmbSizeSelect_3.clear()
        self.ui.cmbSizeSelect_3.setPlaceholderText("구경 필드 선택")
        self.ui.cmbFeatureID_3.clear()
        self.ui.cmbFeatureID_3.setPlaceholderText("관로ID 필드 선택")        
        self.ui.cmbWidthSelect.clear()
        self.ui.cmbWidthSelect.setPlaceholderText("폭 필드 선택")
        self.ui.cmbHightSelect.clear()
        self.ui.cmbHightSelect.setPlaceholderText("높이 필드 선택")
        self.ui.cmbPipeRow.clear()
        self.ui.cmbPipeRow.setPlaceholderText("관열수 필드 선택")
        self.ui.cmbCircleSeg_3.clear()
        self.ui.cmbCircleSeg_3.addItems(["16", "24", "32", "64", "128", "256"])
        self.ui.cmbCircleSeg_3.setPlaceholderText("원형 단면 분할 갯수를 선택하여 주세요")  # 기본값 설정
        self.ui.cmbResamplingIntervals_3.clear()
        self.ui.cmbResamplingIntervals_3.addItems(["0.0","1.0", "2.0", "3.0", "5.0", "10.0", "20.0"])
        self.ui.cmbResamplingIntervals_3.setPlaceholderText("선형 중간점 간격을 설정하여 주세요")
        self.ui.ledConnectTolerance_3.clear()
        self.ui.ledConnectTolerance_3.setPlaceholderText("끝점 연결 허용오차를 입력하여 주세요")
        self.ui.cmbExportFileFormat_3.clear()
        self.ui.cmbExportFileFormat_3.addItems(["OBJ", "GLB", "GLTF"])
        self.ui.cmbExportFileFormat_3.setPlaceholderText("내보낼 파일 형식을 선택하여 주세요")
        self.ui.chkResampleOption_3.setChecked(False)
        self.ui.btnStart_3.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop_3.setEnabled(False)  # 변환중지 버튼 비활성화
        self.ui.lblProgressMsg_3.clear()
        self.ui.lblProgressMsg_3.setText("변환작업 대기 중 ...")
        self.ui.progressBar_3.setValue(0)
        self.ui.progressBar_3.setRange(0, 100)  # 프로그래스바 범위 설정

    def setup_tab4(self):
        # tab3 UI 초기화
        self.ui.leWorkspaceFolder.clear()
        self.ui.leWorkspaceFolder.setPlaceholderText("gpkg 파일이 존재하는 폴더를 선택하여 주세요")
        self.ui.leSourceFolder.clear()
        self.ui.leSourceFolder.setPlaceholderText("원본 Shape/DBF 파일이 있는 폴더를 선택하여 주세요")
        self.ui.lwGpkgFileList.clear()
        self.ui.lwAnalysisFileList.clear()
        self.ui.teAnalysisResults.clear()
        self.ui.btAllFilesAdd.setEnabled(False)
        self.ui.btSelectedFilesAdd.setEnabled(False)
        self.ui.btRemoveSelected.setEnabled(False)
        self.ui.btRemoveAll.setEnabled(False)
        self.ui.btAnalysisWorkStart.setEnabled(False)
        self.ui.btAnalysisWorkStop.setEnabled(False)

    def on_tab_changed(self, index):
        if index == 0:
            self.setup_tab1()
        elif index == 1:
            self.setup_tab2()
        elif index == 2:
            self.setup_tab3()
        elif index == 3:
            self.setup_tab4()

    def set_othertab_none(self):
        # 다른 탭 비활성화
        for i in range(self.ui.tabWidget.count()):
            if i!= self.ui.tabWidget.currentIndex():
                self.ui.tabWidget.setTabEnabled(i, False)
    
    def set_alltab_enable(self):
        # 모든 탭 활성화
        for i in range(self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(i, True)

    # 저장 폴더 선택 버튼 클릭 이벤트 핸들러
    def on_save_folder_browse_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Save Folder", "")
        if folder_path:
            self.ui.ledSaveFolder.setText(folder_path)

    def on_resampling_intervals_changed(self):
        if self.ui.cmbResamplingIntervals.currentText == "0.0":
            if self.ui.chkResampleOption_1.isChecked():
                self.ui.chkResampleOption_1.setChecked(False)
            self.ui.chkResampleOption_1.setEnabled(False)
        else:
            self.ui.chkResampleOption_1.setEnabled(True)
            
    # 닫기 버튼 클릭 이벤트 핸들러
    def on_window_close_clicked(self):
        """
        창 닫기 버튼 클릭 이벤트 핸들러

        이 메서드는 창을 닫을 때 호출됩니다. QApplication의 quit() 메서드를 호출하여
        응용프로그램을 종료합니다.
        """
        QApplication.quit()

    # 파일 선택 버튼 클릭 이벤트 핸들러
    def on_browse_clicked(self):
        """
        Shapefile 선택 버튼 클릭 이벤트 핸들러

        :return: None
        """
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "Shapefile 선택", "", "Shapefiles (*.shp)")
        if file_path:
            if self.ui.tabWidget.currentIndex() == 0:
                self.ui.ledShapeFile.setText(file_path)
                folder_path = os.path.dirname(file_path)
                self.ui.ledSaveFolder.setText(folder_path)
            elif self.ui.tabWidget.currentIndex() == 1:
                sender = self.sender()
                if sender == self.ui.btnBrowse_2:
                    self.ui.ledShapeFile_2.setText(file_path)
                    folder_path = os.path.dirname(file_path)
                    self.ui.ledSaveFolder_2.setText(folder_path)
                elif sender == self.ui.btnBrowsePointshp:
                    self.ui.ledPointShapeFile.setText(file_path)
                    folder_path = os.path.dirname(file_path)
                    self.ui.ledSaveFolder_2.setText(folder_path)
            elif self.ui.tabWidget.currentIndex() == 2:
                self.ui.ledShapeFile_3.setText(file_path)
                folder_path = os.path.dirname(file_path)
                self.ui.ledSaveFolder_3.setText(folder_path)

    # tab_1 적용 버튼 클릭 이벤트 핸들러 
    def on_apply_clicked(self):
        """
        Tab 1의 "적용" 버튼 클릭 이벤트 핸들러

        이 메서드는 Tab 1의 "적용" 버튼 클릭 이벤트를 처리합니다.
        선택한 Shapefile의 DBF 파일을 읽고, DBF 파일의 필드 이름을 콤보박스에 추가합니다.
        또한, 변환 옵션의 기본값을 설정합니다.
        """

        if not self.ui.cmbFacilType.currentText() :
            QMessageBox.warning(self, "Warning", "관로유형을 선택하여 주십시오.")
            return
        if not self.ui.ledShapeFile.text() :
            QMessageBox.warning(self, "Warning", "변환할 shape 파일을 선택하여 주십시오.")
            return
        if not self.ui.ledSaveFolder.text() :
            QMessageBox.warning(self, "Warning", "저장 폴더를 선택하여 주십시오.")
            return

        input_file = self.ui.ledShapeFile.text().strip()
        print("선택한 shape 파일:", input_file)
        if input_file:
            if input_file.endswith(".shp"):
                dbf_file = input_file.replace(".shp", ".dbf")
                print("대상 dbf 파일:", dbf_file)
                try:
                    # DBF 파일을 읽고 처리
                    dbf = DBF(dbf_file)
                    # print("DBF 파일 필드:", dbf.fields)
                    numeric_fields = []
                    character_fields = []
                    # 숫자형/문자형 필드를 구분해서 리스트에 저장
                    for field in dbf.fields:
                        if field.type in ['N', 'F', 'O', 'I']:  # Numeric, Float, Double, Integer types
                            numeric_fields.append(field.name)
                            numeric_fields.sort()  # 숫자형 필드 이름 정렬
                        elif field.type in ['C', 'M']:  # Character, Memo types
                            character_fields.append(field.name)
                            character_fields.sort()  # 문자형 필드 이름 정렬
                    # DBF 파일의 필드 이름을 콤보박스에 추가
                    self.ui.cmbFeatureID.clear()
                    self.ui.cmbDepthSelect.clear()
                    self.ui.cmbSizeSelect.clear()
                    self.ui.cmbFeatureID.addItems(character_fields)
                    self.ui.cmbDepthSelect.addItems(numeric_fields)
                    self.ui.cmbSizeSelect.addItems(numeric_fields)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"DBF 파일을 열 수 없습니다: {e}")
            else:
                QMessageBox.warning(self, "Warning", "선택한 파일이 .shp 파일이 아닙니다.")
        # 변환 옵션 기본값 설정
        self.ui.cmbCircleSeg.setCurrentText("16")  # 원형 단면 분할 갯수 기본값 설정
        self.ui.cmbResamplingIntervals.setCurrentText("0.0")  # 선형 중간점 간격 기본값 설정
        self.ui.chkResampleOption_1.setEnabled(False)  # 리샘플링 옵션 비활성화
        self.ui.ledConnectTolerance.setText("0.02")  # 허용오차 기본값 설정
        self.ui.cmbExportFileFormat.setCurrentText("GLB")  # 내보낼 파일 형식 기본값 설정   
        self.ui.btnStart.setEnabled(True)   # 변환시작 버튼 활성화

    # tab_2 적용 버튼 클릭 이벤트 핸들러 
    def on_apply_mix_clicked(self):
        if not self.ui.cmbFacilType_2.currentText() :
            QMessageBox.warning(self, "Warning", "관로유형을 선택하여 주십시오.")
            return
        if not self.ui.ledShapeFile_2.text() :
            QMessageBox.warning(self, "Warning", "관로 shape 파일을 선택하여 주십시오.")
            return
        if not self.ui.ledPointShapeFile.text() :
            QMessageBox.warning(self, "Warning", "심도 shape 파일을 선택하여 주십시오.")
            return
        if not self.ui.ledSaveFolder_2.text() :
            QMessageBox.warning(self, "Warning", "저장 폴더를 선택하여 주십시오.")
            return
        
        lienshp_file = self.ui.ledShapeFile_2.text().strip()
        pointshp_file = self.ui.ledPointShapeFile.text().strip()
        if lienshp_file and lienshp_file :
            if lienshp_file.endswith(".shp") and pointshp_file.endswith(".shp"):
                line_dbf_file = lienshp_file.replace(".shp", ".dbf")
                point_dbf_file = pointshp_file.replace(".shp", ".dbf")
                try:
                    # DBF 파일을 읽고 처리하는 로직 추가
                    line_dbf = DBF(line_dbf_file)
                    point_dbf = DBF(point_dbf_file)
                    line_numeric_fields = []
                    character_fields = []
                    point_numeric_fields = []
                    # 숫자형/문자형 필드를 구분해서 리스트에 저장
                    for field in line_dbf.fields:
                        if field.type in ['N', 'F', 'O', 'I']:  # Numeric, Float, Double, Integer types
                            line_numeric_fields.append(field.name)
                            line_numeric_fields.sort()  # 숫자형 필드 이름 정렬
                        elif field.type in ['C', 'M']:  # Character, Memo types
                            character_fields.append(field.name)
                            character_fields.sort()  # 문자형 필드 이름 정렬
                    for field in point_dbf.fields:
                        if field.type in ['N', 'F', 'O', 'I']:  # Numeric, Float, Double, Integer types
                            point_numeric_fields.append(field.name)
                            point_numeric_fields.sort()  # 숫자형 필드 이름 정렬

                    # DBF 파일의 필드 이름을 콤보박스에 추가
                    self.ui.cmbAvrgDepthSelect.clear()
                    self.ui.cmbAvrgDepthSelect.addItems(line_numeric_fields)
                    self.ui.cmbSizeSelect_2.clear()
                    self.ui.cmbSizeSelect_2.addItems(line_numeric_fields)
                    self.ui.cmbFeatureID_2.clear()
                    self.ui.cmbFeatureID_2.addItems(character_fields)
                    self.ui.cmbPointDepthSelect.clear()
                    self.ui.cmbPointDepthSelect.addItems(point_numeric_fields)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"DBF 파일을 열 수 없습니다: {e}")
            else:
                QMessageBox.warning(self, "Warning", "선택한 파일이 .shp 파일이 아닙니다.")
        # 변환 옵션 기본값 설정
        self.ui.cmbCircleSeg_2.setCurrentText("16")  # 원형 단면 분할 갯수 기본값 설정
        self.ui.ledConnectTolerance_2.setText("0.02")  # 허용오차 기본값 설정
        self.ui.cmbExportFileFormat_2.setCurrentText("GLB")  # 내보낼 파일 형식 기본값 설정   
        self.ui.btnStart_2.setEnabled(True)   # 변환시작 버튼 활성화

    #tab_2 GDF 내보내기 버튼 클릭 이벤트 핸들러
    def export_gdf(self):
        # 입력값 체크 검사
        if not self.tab2_input_validate():
            print("입력값 검사 실패")
            return
        print("입력값 검사 통과, GDF 내보내기 시작")

        # 대기 커서로 전환
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        # QProgressDialog: 0~100 진행률
        self._progress_dialog = WideProgressDialog("GeoDataFrame 파일 내보내기 중...", None, 0, 100, self, min_width=600)
        self._progress_dialog.setWindowTitle("GeoDataFrame 파일 내보내기")
        self._progress_dialog.setCancelButton(None)             # 취소 버튼 비활성화
        self._progress_dialog.setWindowModality(Qt.WindowModal) # 모달
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)
        self._progress_dialog.show()

        # Worker + QThread 구성
        facil_type = self.ui.cmbFacilType_2.currentText().strip()
        self._export_worker = ExportWorker(
            facil_type=facil_type,
            line_shp=self.ui.ledShapeFile_2.text().strip(),
            depth_shp=self.ui.ledPointShapeFile.text().strip(),
            save_dir=self.ui.ledSaveFolder_2.text().strip(),
            avg_depth_field=self.ui.cmbAvrgDepthSelect.currentText().strip(),
            pipe_size_field=self.ui.cmbSizeSelect_2.currentText().strip(),
            point_depth_field=self.ui.cmbPointDepthSelect.currentText().strip(),
            feature_id_field=self.ui.cmbFeatureID_2.currentText().strip(),
            layer_name=facil_type
        )
        self._export_thread = QThread(self)
        self._export_worker.moveToThread(self._export_thread)
        # 시그널 연결 (진행률/상태 UI 업데이트는 메인 스레드에서 실행됨)
        self._export_thread.started.connect(self._export_worker.run)
        self._export_worker.finished.connect(self._on_finished)
        self._export_worker.error.connect(self._on_error)
        self._export_worker.progress.connect(self._progress_dialog.setValue)
        self._export_worker.status.connect(self._progress_dialog.setLabelText)

        # 시작
        self._export_thread.start()

    # 정리 함수
    def _cleanup(self):
        try:
            QGuiApplication.restoreOverrideCursor()
        except Exception:
            pass
        if getattr(self, "_progress_dialog", None):
            self._progress_dialog.close()
            self._progress_dialog = None
        if getattr(self, "_export_worker", None):
            self._export_worker.deleteLater()
            self._export_worker = None
        if getattr(self, "_export_thread", None):
            self._export_thread.quit()
            self._export_thread.wait()
            self._export_thread.deleteLater()
            self._export_thread = None

    # 콜백
    def _on_finished(self):
        # 다이얼로그가 자동으로 닫히지 않는 경우를 대비해 클린업
        self._cleanup()
        QMessageBox.information(self, "Info", "GDF 파일 내보내기가 완료되었습니다.")

    def _on_error(self, err_msg: str):
        self._cleanup()
        QMessageBox.critical(self, "Error", f"GDF 파일 내보내기 중 오류가 발생했습니다: {err_msg}")

    # tab_3 적용 버튼 클릭 이벤트 핸들러 
    def on_apply_sewerage_clicked(self):
        if not self.ui.cmbFacilType_3.currentText() :
            QMessageBox.warning(self, "Warning", "관로유형을 선택하여 주십시오.")
            return
        if not self.ui.ledShapeFile_3.text() :
            QMessageBox.warning(self, "Warning", "변환할 shape 파일을 선택하여 주십시오.")
            return
        if not self.ui.ledSaveFolder_3.text() :
            QMessageBox.warning(self, "Warning", "저장 폴더를 선택하여 주십시오.")
            return

        input_file = self.ui.ledShapeFile_3.text().strip()
        if input_file:
            if input_file.endswith(".shp"):
                dbf_file = input_file.replace(".shp", ".dbf")
                print("대상 dbf 파일:", dbf_file)
                try:
                    # DBF 파일을 읽고 처리
                    dbf = DBF(dbf_file)
                    # print("DBF 파일 필드:", dbf.fields)
                    numeric_fields = []
                    character_fields = []
                    # 숫자형/문자형 필드를 구분해서 리스트에 저장
                    for field in dbf.fields:
                        if field.type in ['N', 'F', 'O', 'I']:  # Numeric, Float, Double, Integer types
                            numeric_fields.append(field.name)
                            numeric_fields.sort()  # 숫자형 필드 이름 정렬
                        elif field.type in ['C', 'M']:  # Character, Memo types
                            character_fields.append(field.name)
                            character_fields.sort()  # 문자형 필드 이름 정렬
                    # DBF 파일의 필드 이름을 콤보박스에 추가
                    self.ui.cmbStartDepthSelect.clear()
                    self.ui.cmbEndDepthSelect.clear()
                    self.ui.cmbWidthSelect.clear()
                    self.ui.cmbHightSelect.clear()
                    self.ui.cmbSizeSelect_3.clear()
                    self.ui.cmbFeatureID_3.clear()
                    self.ui.cmbPipeRow.clear()
                    self.ui.cmbStartDepthSelect.addItems(numeric_fields)
                    self.ui.cmbStartDepthSelect.setCurrentText("SPT_DEP")  # 시작심도필드 기본값 설정
                    self.ui.cmbEndDepthSelect.addItems(numeric_fields)
                    self.ui.cmbEndDepthSelect.setCurrentText("EPT_DEP")    # 종료심도필드 기본값 설정
                    self.ui.cmbWidthSelect.addItems(numeric_fields)
                    self.ui.cmbWidthSelect.setCurrentText("HOL_STD")        # 폭필드 기본값 설정
                    self.ui.cmbHightSelect.addItems(numeric_fields)
                    self.ui.cmbHightSelect.setCurrentText("VEL_STD")        # 높이필드 기본값 설정
                    self.ui.cmbSizeSelect_3.addItems(numeric_fields)
                    self.ui.cmbSizeSelect_3.setCurrentText("OM_PE_DTR")      # 구경필드 기본값 설정
                    self.ui.cmbFeatureID_3.addItems(character_fields)
                    self.ui.cmbFeatureID_3.setCurrentText("IDN")      # 관로ID필드 기본값 설정  
                    self.ui.cmbPipeRow.addItems(numeric_fields)
                    self.ui.cmbPipeRow.setCurrentText("PE_RW_NUM")  # 관열수필드 기본값 설정                  
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"DBF 파일을 열 수 없습니다: {e}")
            else:
                QMessageBox.warning(self, "Warning", "선택한 파일이 .shp 파일이 아닙니다.")
        # 변환 옵션 기본값 설정
        self.ui.cmbCircleSeg_3.setCurrentText("16")  # 원형 단면 분할 갯수 기본값 설정
        self.ui.cmbResamplingIntervals_3.setCurrentText("0.0")  # 선형 중간점 간격 기본값 설정
        self.ui.ledConnectTolerance_3.setText("0.02")  # 허용오차 기본값 설정
        self.ui.cmbExportFileFormat_3.setCurrentText("GLB")  # 내보낼 파일 형식 기본값 설정   
        self.ui.btnStart_3.setEnabled(True)   # 변환시작 버튼 활성화

    """
    >>>>>>>>>>>>>> 평균심도방식 변환 프로세스 <<<<<<<<<<<<
    """
    def start_avrg_conversion(self):
        # 변환작업 중 상태 플래그 설정
        self.is_working = True 
        self.set_othertab_none()  # 다른 탭 비활성화
        # 입력값 체크 검사
        input_data = self.tab1_input_validate()

        if input_data is None:
            return        
        print("입력값 검사 통과, 변환 시작 준비")
        #스레드 running 상태 검사
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "이전 작업이 아직 종료되지 않았습니다. 현재 작업 취소하고 새로 시작할 수 있습니다.")
            self.cancel_avrg_conversion()
            return
        self.was_canceled = False

        print("worker 객체 생성과 스레드 시작")
        # 1. 스레드와 워커 객체 생성
        self.worker_thread = QtCore.QThread()
        self.worker = ConversionWorker(**input_data)    # 필요한 인자 전달
        # 2. 워커를 스레드로 이동
        self.worker.moveToThread(self.worker_thread)
        # 3. 스레드 시작 시 워커의 run 메서드 호출
        self.worker_thread.started.connect(self.worker.run)
        # 4. 워커의 시그널을 메인 윈도우의 슬롯에 연결
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self._on_error)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_stopped)
        # 54. 스레드 시작
        self.worker_thread.start()
        print("스레드 시작 완료")
        self.ui.btnStart.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop.setEnabled(True)  # 변환중지 버튼 비활성화
        self.ui.progressBar.setValue(0)  # 프로그래스바 초기화

    # tab_1 입력값 검사
    def tab1_input_validate(self):
        # 입력값 입력 누락 검사
        if not self.ui.ledShapeFile.text():
            QMessageBox.warning(self, "Warning", "변환할 shape 파일을 선택하여 주십시오.")
            return
        if not self.ui.cmbFeatureID.currentText():
            QMessageBox.warning(self, "Warning", "관로ID 필드를 선택하여 주십시오.")
            return
        if not self.ui.cmbDepthSelect.currentText():
            QMessageBox.warning(self, "Warning", "평균심도 필드를 선택하여 주십시오.")
            return
        if not self.ui.cmbSizeSelect.currentText():
            QMessageBox.warning(self, "Warning", "구경 크기 필드를 선택하여 주십시오.")
            return
        if not self.ui.cmbCircleSeg.currentText():
            QMessageBox.warning(self, "Warning", "원형 단면 분할 갯수를 선택하여 주십시오.")
            return
        if not self.ui.cmbResamplingIntervals.currentText():
            QMessageBox.warning(self, "Warning", "선형 중간점 간격을 설정하여 주십시오.")
            return
        if not self.ui.ledConnectTolerance.text():
            QMessageBox.warning(self, "Warning", "끝점 연결 허용오차를 입력하여 주십시오.")
            return
        if not self.ui.cmbExportFileFormat.currentText():
            QMessageBox.warning(self, "Warning", "내보낼 파일 형식을 선택하여 주십시오.")
            return

        shp_file = self.ui.ledShapeFile.text().strip()
        output_folder = self.ui.ledSaveFolder.text().strip()
        feature_id = self.ui.cmbFeatureID.currentText().upper()
        depth_f = self.ui.cmbDepthSelect.currentText().upper()  # 평균심도 필드
        size_f = self.ui.cmbSizeSelect.currentText().upper()  # 구경 크기 필드
    
        if not shp_file or not os.path.exists(shp_file):
            QMessageBox.warning(self, "입력 오류", "올바른 Shapefile(.shp)을 선택하세요.")
            return None
        if not output_folder or not os.path.isdir(output_folder):
            QMessageBox.warning(self, "입력 오류", "올바른 출력 폴더를 선택하세요.")
            return None
        try:
            tol = float(self.ui.ledConnectTolerance.text().strip())
        except Exception:
            QMessageBox.warning(self, "입력 오류", "허용오차 입력값은 숫자여야 합니다.")
            return None
        print("입력값 검사 완료>>>>>>>>>>>")
        return {
            'facility_type': self.ui.cmbFacilType.currentText(),
            'shapefile': shp_file,
            'output_folder': output_folder,
            'feature_id_field': feature_id,
            'avrg_depth_field': depth_f,
            'pipe_size_field': size_f,
            'circle_sections': int(self.ui.cmbCircleSeg.currentText()),
            'resample_interval': float(self.ui.cmbResamplingIntervals.currentText()),
            'export_format': self.ui.cmbExportFileFormat.currentText().lower(),
            'tolerance': tol,
            'preserve_vertices': self.ui.chkResampleOption_1.isChecked()
        }

    # 평균심도방식 변환작업 취소   
    def cancel_avrg_conversion(self):
        # 1. 현재 스레드가 존재하고 실행 중인지 확인
        if self.worker_thread is None or not self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "현재 실행 중인 변환 작업이 없습니다.")
        else:
        # 2. 워커에게 취소 요청
            print("변환 작업 취소 요청")
            self.ui.lblProgressMsg.setText("변환 작업 중지 중... 잠시만 기다려주세요.")
            try:
                # 커서 변경
                QGuiApplication.setOverrideCursor(Qt.WaitCursor)
                self.worker.request_cancel()
                # 취소가 완료되면 finished 시그널이 발생하므로 여기선 대기만 함
                self.worker_thread.quit()  # 현재 스레드 종료
                self.worker_thread.wait()  # 스레드가 종료될 때까지 대기
                # 워커와 스레드 객체를 정리
                self.worker.finished.disconnect(self._on_conversion_finished)
                self.worker = None
                self.worker_thread = None
                print("워커와 스레드 객체 정리 완료")              
                self.was_canceled = True
            except Exception as e:
                print(f"변환 작업 취소 중 오류 발생: {e}")
            finally:
                # 커서 복구
                QGuiApplication.restoreOverrideCursor()
                self.is_working = False # 작업 중 상태 플래그 해제
                self.set_alltab_enable()  # 모든 탭 활성화
                print("변환 작업 취소 완료")

        self.ui.progressBar.setValue(0)  # 프로그래스바 초기화
        self.ui.btnStop.setEnabled(False)
        self.ui.btnStart.setEnabled(True)  # 변환시작 버튼 활성화
        self.ui.lblProgressMsg.setText("변환 작업이 취소되었습니다. 다시 시작할 수 있습니다.")

    """     
    >>>>>>>>>>>>>> 포인트심도 결합방식 변환 프로세스 <<<<<<<<<<<<    
    """
    # 심도포인트 결합방식 변환 프로세스 시작
    def start_mixed_conversion(self):
        # 변환작업 중 상태 플래그 설정
        self.is_working = True 
        self.set_othertab_none()  # 다른 탭 비활성화
        # 입력값 체크 검사
        if not self.tab2_input_validate():
            print("입력값 검사 실패")
            return
        # mixed_converter 객체에 전달할 파라미터 셋팅
        input_data = dict(
            facility_type=self.ui.cmbFacilType_2.currentText(),
            line_shp=self.ui.ledShapeFile_2.text().strip(),
            point_shp=self.ui.ledPointShapeFile.text().strip(),
            output_folder=self.ui.ledSaveFolder_2.text().strip(),
            feature_id_field=self.ui.cmbFeatureID_2.currentText().upper(),
            avrg_depth_field=self.ui.cmbAvrgDepthSelect.currentText().upper(),
            pipe_size_field=self.ui.cmbSizeSelect_2.currentText().upper(),
            point_depth_field=self.ui.cmbPointDepthSelect.currentText().upper(),
            circle_sections=int(self.ui.cmbCircleSeg_2.currentText()),
            tolerance=float(self.ui.ledConnectTolerance_2.text().strip()),
            export_format=self.ui.cmbExportFileFormat_2.currentText().strip().lower()
        )

        if input_data is None:
            return        
        print("입력값 검사 통과, 변환 시작 준비")
        
        #스레드 running 상태 검사
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "이전 작업이 아직 종료되지 않았습니다. 현재 작업 취소하고 새로 시작할 수 있습니다.")
            self.cancel_mixed_conversion()
            return
        self.was_canceled = False

        print("worker 객체 생성과 스레드 시작")

        # 1. 스레드와 워커 객체 생성
        self.worker_thread = QtCore.QThread()
        self.worker = MixedConverter(**input_data)    # 필요한 인자 전달
        # 2. 워커를 스레드로 이동
        self.worker.moveToThread(self.worker_thread)
        # 3. 스레드 시작 시 워커의 run 메서드 호출
        self.worker_thread.started.connect(self.worker.run)
        # 4. 워커의 시그널을 메인 윈도우의 슬롯에 연결
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self._on_error)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_stopped)
        # 5. 스레드 시작
        self.worker_thread.start()
        print("스레드 시작 완료")# 5. 변환 시작 UI 업데이트
        self.ui.btnStart_2.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop_2.setEnabled(True)  # 변환중지 버튼 비활성화
        self.ui.progressBar_2.setValue(0)

    # tab_2 입력값 검사
    def tab2_input_validate(self):
        def err(msg):
            QMessageBox.warning(self, "Validation", msg)
            return False

        line_shp = self.ui.ledShapeFile_2.text().strip()
        point_shp = self.ui.ledPointShapeFile.text().strip()
        out_folder = self.ui.ledSaveFolder_2.text().strip()
        if not line_shp or not os.path.exists(line_shp):
            return err("선형 SHP 파일이 없습니다.")
        if not point_shp or not os.path.exists(point_shp):
            return err("포인트 SHP 파일이 없습니다.")
        if not out_folder or not os.path.isdir(out_folder):
            return err("출력 폴더가 없습니다.")
        if not self.ui.cmbFeatureID_2.currentText().strip():
            return err("관로ID 필드명이 필요합니다.")
        if not self.ui.cmbAvrgDepthSelect.currentText().strip():
            return err("평균심도 필드명이 필요합니다.")
        if not self.ui.cmbSizeSelect_2.currentText().strip():
            return err("단면크기(구경) 필드명이 필요합니다.")
        if not self.ui.cmbPointDepthSelect.currentText().strip():
            return err("포인트 심도 필드명이 필요합니다.")
        if not self.ui.cmbCircleSeg_2.currentText():
            return err("원형 단면 분할 갯수를 선택하여 주십시오.")
        if not self.ui.ledConnectTolerance_2.text().strip():
            return err("끝점 연결 허용오차를 입력하여 주십시오.")
        try:
            float(self.ui.ledConnectTolerance_2.text().strip())
        except Exception:
            return err("허용오차 입력값은 숫자여야 합니다.")
        if not self.ui.cmbExportFileFormat_2.currentText():
            return err("내보낼 파일 형식을 선택하여 주십시오.")
        return True

    def cancel_mixed_conversion(self):
        # 1. 현재 스레드가 존재하고 실행 중인지 확인
        if self.worker_thread is None or not self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "현재 실행 중인 스레드가 없습니다.")
        else:
        # 2. 워커에게 취소 요청
            print("변환 작업 취소 요청")
            self.ui.lblProgressMsg_2.setText("변환 작업 중지 중... 잠시만 기다려주세요.")
            try:
                # 커서 변경
                QGuiApplication.setOverrideCursor(Qt.WaitCursor)
                self.worker.request_cancel()
                # 취소가 완료되면 finished 시그널이 발생하므로 여기선 대기만 함
                self.worker_thread.quit()  # 현재 스레드 종료
                self.worker_thread.wait()  # 스레드가 종료될 때까지 대기
                self.worker.finished.disconnect(self._on_conversion_finished)
                self.worker = None
                self.worker_thread = None
                self.was_canceled = True
                print("워커와 스레드 객체 정리 완료")     
            except Exception as e:
                print(f"변환 작업 취소 중 오류 발생: {e}")
            finally:
                # 커서 복구
                QGuiApplication.restoreOverrideCursor()
                self.is_working = False # 작업 중 상태 플래그 해제
                self.set_alltab_enable()  # 모든 탭 활성화
                print("변환 작업 취소 완료")

        self.ui.progressBar_2.setValue(0)  # 프로그래스바 초기화
        self.ui.btnStop_2.setEnabled(False)
        self.ui.btnStart_2.setEnabled(True)  # 변환시작 버튼 활성화
        self.ui.lblProgressMsg_2.setText("변환 작업이 취소되었습니다. 다시 시작할 수 있습니다.")

    """
    >>>>>>>>>>>>>> 하수(원형+박스형 혼합)방식 변환 프로세스 <<<<<<<<<<<<
    """
    def start_sewerage_conversion(self):
        # 변환작업 중 상태 플래그 설정
        self.is_working = True 
        self.set_othertab_none()  # 다른 탭 비활성화
        # 입력값 체크 검사
        if not self.tab3_input_validate():
            print("입력값 검사 실패")
            return
        # mixed_converter 객체에 전달할 파라미터 셋팅
        input_data = dict(
            facility_type=self.ui.cmbFacilType_3.currentText(),
            line_shp=self.ui.ledShapeFile_3.text().strip(),
            output_folder=self.ui.ledSaveFolder_3.text().strip(),
            start_depth_field=self.ui.cmbStartDepthSelect.currentText().upper(),
            end_depth_field=self.ui.cmbEndDepthSelect.currentText().upper(),
            pipe_size_field=self.ui.cmbSizeSelect_3.currentText().upper(),
            hol_width_field=self.ui.cmbWidthSelect.currentText().upper(),
            ver_len_field=self.ui.cmbHightSelect.currentText().upper(),
            feature_id_field=self.ui.cmbFeatureID_3.currentText().upper(),
            pipe_row_field=self.ui.cmbPipeRow.currentText().upper(),
            circle_seg=int(self.ui.cmbCircleSeg_3.currentText()),
            intervals=float(self.ui.cmbResamplingIntervals_3.currentText()),
            tolerance=float(self.ui.ledConnectTolerance_3.text()),
            export_format=self.ui.cmbExportFileFormat_3.currentText().strip().lower(),
        )
        if input_data is None:
            return        
        print("입력값 검사 통과, 변환 시작 준비")
        #스레드 running 상태 검사
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "이전 작업이 아직 종료되지 않았습니다. 현재 작업 취소하고 새로 시작할 수 있습니다.")
            self.cancel_avrg_conversion()
            return
        self.was_canceled = False

        print("worker 객체 생성과 스레드 시작")
        # 1. 스레드와 워커 객체 생성
        self.worker_thread = QtCore.QThread()
        self.worker = SewerageConverter(**input_data)    # 필요한 인자 전달
        # 2. 워커를 스레드로 이동
        self.worker.moveToThread(self.worker_thread)
        # 3. 스레드 시작 시 워커의 run 메서드 호출
        self.worker_thread.started.connect(self.worker.run)
        # 4. 워커의 시그널을 메인 윈도우의 슬롯에 연결
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.message.connect(self._on_message)
        self.worker.finished.connect(self._on_conversion_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self._on_error)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._on_thread_stopped)
        # 54. 스레드 시작
        self.worker_thread.start()
        print("스레드 시작 완료")
        self.ui.btnStart_3.setEnabled(False)  # 변환시작 버튼 비활성화
        self.ui.btnStop_3.setEnabled(True)  # 변환중지 버튼 비활성화
        self.ui.progressBar_3.setValue(0)  # 프로그래스바 초기화

    # tab_3 입력값 검사
    def tab3_input_validate(self):
        def err(msg):
            QMessageBox.warning(self, "Validation", msg)
            return False

        line_shp = self.ui.ledShapeFile_3.text().strip()
        out_folder = self.ui.ledSaveFolder_3.text().strip()
        if not line_shp or not os.path.exists(line_shp):
            return err("선형 SHP 파일이 없습니다.")
        if not out_folder or not os.path.isdir(out_folder):
            return err("출력 폴더가 없습니다.")

        if not self.ui.cmbStartDepthSelect.currentText().strip():
            return err("시점심도 필드명을 선택하여 주십시오.")
        if not self.ui.cmbEndDepthSelect.currentText().strip():
            return err("종점심도 필드명을 선택하여 주십시오.")
        if not self.ui.cmbSizeSelect_3.currentText().strip():
            return err("단면크기(구경) 필드명을 선택하여 주십시오.")
        if not self.ui.cmbWidthSelect.currentText().strip():
            return err("가로길이 필드명을 선택하여 주십시오.")
        if not self.ui.cmbHightSelect.currentText().strip():
            return err("세로길이 필드명을 선택하여 주십시오.")
        if not self.ui.cmbFeatureID_3.currentText().strip():
            return err("관로ID 필드명을 선택하여 주십시오.")
        if not self.ui.cmbPipeRow.currentText().strip():
            return err("관로 열수 필드명을 선택하여 주십시오.")
        if not self.ui.cmbCircleSeg_3.currentText():
            return err("원형 단면 분할 갯수를 선택하여 주십시오.")
        if not self.ui.cmbResamplingIntervals_3.currentText():
            return err("선형 중간점 간격을 선택하여 주십시오.")
        if not self.ui.ledConnectTolerance_3.text().strip():
            return err("끝점 연결 허용오차를 입력하여 주십시오.")
        try:
            float(self.ui.ledConnectTolerance_3.text().strip())
        except Exception:
            return err("허용오차 입력값은 숫자여야 합니다.")
        if not self.ui.cmbExportFileFormat_3.currentText():
            return err("내보낼 파일 형식을 선택하여 주십시오.")
        return True

    # 하수컨버터 변환작업 취소   
    def cancel_sewerage_conversion(self):
        # 1. 현재 스레드가 존재하고 실행 중인지 확인
        if self.worker_thread is None or not self.worker_thread.isRunning():
            QMessageBox.warning(self, "Warning", "현재 실행 중인 변환 작업이 없습니다.")
        else:
        # 2. 워커에게 취소 요청
            print("변환 작업 취소 요청")
            self.ui.lblProgressMsg_3.setText("변환 작업 중지 중... 잠시만 기다려주세요.")
            try:
                # 커서 변경
                QGuiApplication.setOverrideCursor(Qt.WaitCursor)
                self.worker.request_cancel()
                # 취소가 완료되면 finished 시그널이 발생하므로 여기선 대기만 함
                self.worker_thread.quit()  # 현재 스레드 종료
                self.worker_thread.wait()  # 스레드가 종료될 때까지 대기
                # 워커와 스레드 객체를 정리
                self.worker.finished.disconnect(self._on_conversion_finished)
                self.worker = None
                self.worker_thread = None
                print("워커와 스레드 객체 정리 완료")              
                self.was_canceled = True
            except Exception as e:
                print(f"변환 작업 취소 중 오류 발생: {e}")
            finally:
                # 커서 복구
                QGuiApplication.restoreOverrideCursor()
                self.is_working = False # 작업 중 상태 플래그 해제
                self.set_alltab_enable()  # 모든 탭 활성화
                print("변환 작업 취소 완료")

        self.ui.progressBar_3.setValue(0)  # 프로그래스바 초기화
        self.ui.btnStop_3.setEnabled(False)
        self.ui.btnStart_3.setEnabled(True)  # 변환시작 버튼 활성화
        self.ui.lblProgressMsg_3.setText("변환 작업이 취소되었습니다. 다시 시작할 수 있습니다.")

    @Slot(bool)
    def _on_conversion_finished(self, success: bool):
        self.was_canceled = (not success)
        # 스레드가 작업을 마치면 종료
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.is_working = False # 작업 중 상태 플래그 해제
        self.set_alltab_enable()  # 모든 탭 활성화
        if self.was_canceled:
            # 취소된 경우, 재시작 버튼 활성화
            QMessageBox.information(self, "canceled", "작업이 취소되었습니다. 재시작할 수 있습니다.")
        else:
            # 정상 종료된 경우
            if self.ui.tabWidget.currentIndex() == 0:
                QMessageBox.information(self, "Success", f"변환 작업이 완료되었습니다.\n {self.ui.ledSaveFolder.text()} 폴더에 저장되었습니다.")
                self.ui.lblProgressMsg.setText("변환 작업 대기 중 ...")
                self.ui.btnStart.setEnabled(True) # 새 작업을 위해 시작 버튼 활성화
                self.ui.btnStop.setEnabled(False) # 변환중지 버튼 비활성화
                self.ui.progressBar.setValue(0)  # 프로그래스바 초기화
            elif self.ui.tabWidget.currentIndex() == 1:
                QMessageBox.information(self, "Success", f"변환 작업이 완료되었습니다.\n {self.ui.ledSaveFolder_2.text()} 폴더에 저장되었습니다.")
                self.ui.lblProgressMsg_2.setText("변환 작업 대기 중 ...")
                self.ui.btnStart_2.setEnabled(True) # 새 작업을 위해 시작 버튼 활성화
                self.ui.btnStop_2.setEnabled(False) # 변환중지 버튼 비활성화
                self.ui.progressBar_2.setValue(0)  # 프로그래스바 초기화
            elif self.ui.tabWidget.currentIndex() == 2:
                QMessageBox.information(self, "Success", f"변환 작업이 완료되었습니다.\n {self.ui.ledSaveFolder_3.text()} 폴더에 저장되었습니다.")
                self.ui.lblProgressMsg_3.setText("변환 작업 대기 중 ...")
                self.ui.btnStart_3.setEnabled(True) # 새 작업을 위해 시작 버튼 활성화
                self.ui.btnStop_3.setEnabled(False) # 변환중지 버튼 비활성화
                self.ui.progressBar_3.setValue(0)  # 프로그래스바 초기화

    # tab_4 작업 폴더 선택 버튼 클릭 이벤트 핸들러
    def on_work_folder_browse_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Workspace Folder", "")
        if folder_path:
            self.ui.leWorkspaceFolder.setText(folder_path)
            # GUI 초기화
            self.ui.lwGpkgFileList.clear()
            self.ui.lwAnalysisFileList.clear()
            self.ui.teAnalysisResults.clear()
            self.ui.btAllFilesAdd.setEnabled(False)
            self.ui.btSelectedFilesAdd.setEnabled(False)
            self.ui.btRemoveSelected.setEnabled(False)
            self.ui.btRemoveAll.setEnabled(False)
            self.ui.btAnalysisWorkStart.setEnabled(False)
            self.ui.btAnalysisWorkStop.setEnabled(False)

    # tab_4 원본 폴더 선택 버튼 클릭 이벤트 핸들러
    def on_source_folder_browse_clicked(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Source Folder", "")
        if folder_path:
            self.ui.leSourceFolder.setText(folder_path)

    # tab_4 gpkg 파일 목록 불러오기 버튼 클릭 이벤트 핸들러
    def on_Gdf_List_Open_clicked(self):
        folder_path = self.ui.leWorkspaceFolder.text().strip()
        if folder_path:
            self.ui.lwGpkgFileList.clear()
            # 폴더 내의 .gpkg 파일 목록을 리스트 위젯에 추가
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".gpkg"):
                    self.ui.lwGpkgFileList.addItem(file_name)
                    self.ui.lwGpkgFileList.sortItems()
            if self.ui.lwGpkgFileList.count() > 0:
                self.ui.btAllFilesAdd.setEnabled(True)
                self.ui.btSelectedFilesAdd.setEnabled(True)
            else:
                 QMessageBox.warning(self, "Warning", "Workspace Folder에 gpkg 파일이 없습니다. 확인하여 주세요.")
        else:
            QMessageBox.warning(self, "Warning", "gpkg 파일이 존재하는 폴더를 선택하여 주세요.")

    # tab_4 선택한 파일 전체 추가 버튼 클릭 이벤트 핸들러
    def on_add_all_files_clicked(self):
        self.ui.lwAnalysisFileList.clear()
        for index in range(self.ui.lwGpkgFileList.count()):
            item = self.ui.lwGpkgFileList.item(index)
            self.ui.lwAnalysisFileList.addItem(item.text())
        if self.ui.lwAnalysisFileList.count() > 0:
            self.ui.btRemoveSelected.setEnabled(True)
            self.ui.btRemoveAll.setEnabled(True)
        if self.ui.lwAnalysisFileList.count() > 1:
            self.ui.btAnalysisWorkStart.setEnabled(True)
        self.ui.lwGpkgFileList.clear()
        self.ui.btAllFilesAdd.setEnabled(False)
        self.ui.btSelectedFilesAdd.setEnabled(False)

    # tab_4 선택한 파일 선택 추가 버튼 클릭 이벤트 핸들러
    def on_add_selected_files_clicked(self):
        selected_items = self.ui.lwGpkgFileList.selectedItems()
        for item in selected_items:
            self.ui.lwAnalysisFileList.addItem(item.text())
            self.ui.lwGpkgFileList.takeItem(self.ui.lwGpkgFileList.row(item))
        if self.ui.lwAnalysisFileList.count() > 0:
            self.ui.btRemoveSelected.setEnabled(True)
            self.ui.btRemoveAll.setEnabled(True)
        if self.ui.lwAnalysisFileList.count() > 1:
            self.ui.btAnalysisWorkStart.setEnabled(True)
        if self.ui.lwGpkgFileList.count() == 0:
            self.ui.btAllFilesAdd.setEnabled(False)
            self.ui.btSelectedFilesAdd.setEnabled(False)
        
    # tab_4 선택한 파일 선택 삭제 버튼 클릭 이벤트 핸들러
    def on_remove_selected_files_clicked(self):
        selected_items = self.ui.lwAnalysisFileList.selectedItems()
        for item in selected_items:
            self.ui.lwAnalysisFileList.takeItem(self.ui.lwAnalysisFileList.row(item))
            self.ui.lwGpkgFileList.addItem(item.text())
            self.ui.lwGpkgFileList.sortItems()
        if self.ui.lwAnalysisFileList.count() == 0:
            self.ui.btRemoveSelected.setEnabled(False)
            self.ui.btRemoveAll.setEnabled(False)
            self.ui.btAnalysisWorkStart.setEnabled(False)
        if self.ui.lwGpkgFileList.count() > 0:
            self.ui.btAllFilesAdd.setEnabled(True)
            self.ui.btSelectedFilesAdd.setEnabled(True)

    # tab_4 전체 파일 삭제 버튼 클릭 이벤트 핸들러
    def on_remove_all_clicked(self):
        self.ui.lwAnalysisFileList.clear()
        self.ui.btRemoveSelected.setEnabled(False)
        self.ui.btRemoveAll.setEnabled(False)
        self.ui.btAnalysisWorkStart.setEnabled(False)
        self.on_Gdf_List_Open_clicked()

    # tab_4 배치 분석 작업 시작 버튼 클릭 이벤트 핸들러
    def start_batch_analysis(self):
        pass
    
    # tab_4 배치 분석 작업 중지 버튼 클릭 이벤트 핸들러
    def stop_batch_analysis(self):
        pass

    ################################################################################
    # 스레드 및 워커 관련 슬롯들
    ################################################################################
    @Slot()
    def _on_thread_stopped(self):
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.btnStart.setEnabled(True)
            self.ui.btnStop.setEnabled(False)
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.btnStart_2.setEnabled(True)
            self.ui.btnStop_2.setEnabled(False)
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.btnStart_3.setEnabled(True)
            self.ui.btnStop_3.setEnabled(False)

        self.worker_thread = None
        self.worker = None
        self.was_canceled = None

    @Slot(int)
    def _on_progress(self, pct):
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.progressBar.setValue(max(0, min(100, pct)))
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.progressBar_2.setValue(max(0, min(100, pct)))
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.progressBar_3.setValue(max(0, min(100, pct)))

    @Slot(str)
    def _on_message(self, msg):
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.lblProgressMsg.setText(msg)
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.lblProgressMsg_2.setText(msg)
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.lblProgressMsg_3.setText(msg)

    @Slot(str)
    def _on_error(self, err):
        if self.ui.tabWidget.currentIndex() == 0:
            self.ui.btnStart.setEnabled(True)
            self.ui.btnStop.setEnabled(False)
        elif self.ui.tabWidget.currentIndex() == 1:
            self.ui.btnStart_2.setEnabled(True)
            self.ui.btnStop_2.setEnabled(False)
        elif self.ui.tabWidget.currentIndex() == 2:
            self.ui.btnStart_3.setEnabled(True)
            self.ui.btnStop_3.setEnabled(False)
        QMessageBox.critical(self, "에러", err)
        self.worker_thread = None
        self.worker = None
        self.was_canceled = None

    # 창 닫기 이벤트 핸들러
    def _closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            if self.worker is not None:
                self.worker.request_cancel()
            self.worker_thread.quit()
            self.worker_thread.wait()
            self.worker_thread = None
            self.worker = None
            self.was_canceled = None
        event.accept()

def main():
    app = QApplication([])

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()



