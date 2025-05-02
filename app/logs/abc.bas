'==================================
' タイトル：ABC業者の項目表を作成
'==================================
' 目的   : 受入データを集計し、カレンダー形式で重量・金額を記録する
' 引数   : なし
' 返り値 : なし
' 注意点 :
'   - 受入データが正しく入力フォルダに存在しているかを事前に確認する必要がある
'   - 受入データは「kg」単位のデータのみが対象となる
'   - 初回実行時は履歴を考慮して初期処理を行う
'   - データがない場合は警告を表示し処理を終了する
'==================================


'ABC業者の項目表を作成する。
Sub Avel_Summary_Table()
    Dim i As Long
    Dim ii As Long
    Dim j As Long
    Dim jj As Long
    '履歴により最初の処理をする
    If ws_Output.Cells(Ref_sh.hist_check, 2) = "" Then
        Call Average_Summary_Table_fisrt
    End If

    Dim fileName As String
    fileName = "受入一覧"

    Dim display As String
    '入力フォルダの確認
    If Not return_filename_to_headers(ws_Input) = SG_sh.受入 Then
        display = display_FileErrorMessage("CSV読み込みエラー", "【" & fileName & "】", "入力フォルダ")
        MsgBox display, vbExclamation, "警告"
        Exit Sub
    End If

    '===インプットブックからの情報格納=================
    '日付順にシートをソートする
    Call SortrowByColumn(ws_Input, 受入一覧.伝票日付)


    'インプットデータを配列に格納
    Dim Input_ShipDisp As Variant
    Dim input_data As Variant


    With ws_Input.Range("A1").CurrentRegion
        Input_ShipDisp =    .value
    End With

    '日付の不要な（）を削除する
    Input_ShipDisp = Remove_kakkoFromArray(Input_ShipDisp, 受入一覧.伝票日付)


    '最初月内に入る配列にredim
    Dim temp As Variant
    With ws_Output.Cells(Ref_sh.First_Month, 2)
        temp = FilterArrayByDate(Input_ShipDisp, 受入一覧.伝票日付, .value, GetLastDayOfMonth(.value))
        ' 最初月が空の場合の処理
        If IsEmptyArray(temp) Then
            MsgBox Display_NoDate(.value, fileName), vbExclamation, "警告"
            Exit Sub
        End If
    End With

    input_data = temp

    SafeEraseArray (temp)
    SafeEraseArray (Input_ShipDisp)


    If IsError Then: Exit Sub                    'エラーの場合に抜ける

    '===アウトプット情報の格納===================

    Dim ws_Output_Template As Worksheet
    Dim ws_Output_Cal As Worksheet

    Set ws_Output_Template = wb_Output.Worksheets("テンプレート")
    Set ws_Output_Cal = wb_Output.Worksheets("カレンダー_品目")

    ''===カレンダー_品目を埋める=================
    Dim OS_ShipDist As Variant
    With ws_Output_Cal.Range("A1").CurrentRegion
        OS_ShipDist = .value
    End With

    Dim hist_check As Boolean
    Dim dim_num As Long
    Dim initialized As Boolean

    Dim OS_Count As Variant
    With wb_Output.Worksheets("カレンダー_台数").Range("A1").CurrentRegion
        OS_Count = .value
    End With

    Dim culnum As Long

    For i = 4 To UBound(OS_ShipDist, 1)          ' カレンダーでループ
        initialized = False
        '台数：ユニーク数を計算する
        temp = FilterArrayByDate(input_data, 受入一覧.伝票日付, CDate(OS_ShipDist(i, 2)), CDate(OS_ShipDist(i, 2)))

        '空データなら次のi
        If Not IsArray(temp) Or UBound(temp, 1) < LBound(temp, 1) Then
            GoTo next_loop
        End If


        '初期化
        If initialized = False Then
            For jj = 3 To UBound(OS_ShipDist, 2)
                OS_ShipDist(i, jj) = 0
            Next jj

            For jj = 3 To UBound(OS_Count, 2)
                OS_Count(i - 1, jj) = 0          'iはマイナス１
            Next jj
            initialized = True
        End If


        Dim sum_kg_2D As Variant
        Dim sum_kg_arr As Variant
        Dim sum_uriage_2D As Variant
        Dim sum_uriage_arr As Variant
        Dim temp_sum As Variant

        '総品目合計

        temp_sum = FilterArray_ByKeyword(temp, 受入一覧.単位名, "kg")
        temp_sum = FilterArray_ByKeyword(temp_sum, 受入一覧.伝票区分名, "売上")

        sum_kg_2D = col1D_from2DArray(temp_sum, 受入一覧.正味重量)
        sum_kg_arr = SumArray(sum_kg_2D)

        sum_uriage_2D = col1D_from2DArray(temp_sum, 受入一覧.金額)
        sum_uriage_arr = SumArray(sum_uriage_2D)

        OS_ShipDist(i, AS_item_cul.合計kg) = sum_kg_arr
        OS_ShipDist(i, AS_item_cul.合計売上) = sum_uriage_arr



        'インプットデータのループ
        culnum = 1
        Do While culnum <= UBound(temp, 1)


            ' ABC項目の計算
            For j = 5 To UBound(OS_ShipDist, 2) - 1 Step 2 ' 2ステップごとに進む

                ' 変数に一時保存（可読性向上）
                Dim 集計CD As String, 品名 As String, 単位名 As String, 伝票区分名 As String
                集計CD = temp(culnum, 受入一覧.集計項目CD)
                品名 = temp(culnum, 受入一覧.品名)
                単位名 = temp(culnum, 受入一覧.単位名)
                伝票区分名 = temp(culnum, 受入一覧.伝票区分名)

                ' 条件を1つのIfでまとめる
                If 集計CD = OS_ShipDist(1, j) And _
                   品名 = OS_ShipDist(2, j) And _
                   単位名 = "kg" And _
                   伝票区分名 = "売上" Then

                    ' 計算処理
                    OS_ShipDist(i, j) = OS_ShipDist(i, j) + temp(culnum, 受入一覧.正味重量)
                    OS_ShipDist(i, j + 1) = OS_ShipDist(i, j + 1) + temp(culnum, 受入一覧.金額)

                    ' マッチしたら即終了（無駄なループを減らす）
                    Exit For
                End If

            Next j



            'さらにABCでソートして、台数、重量を算出する。
            Dim tempabc_cul As Long
            tempabc_cul = 1

            For ii = 1 To 6
                Dim temp_abc As Variant
                Dim car_sum As Long
                Dim height_arr As Variant
                Dim height_sum As Long



                ' 各ABC毎の配列を作成
                temp_abc = FilterArray_ByKeyword(temp, 受入一覧.集計項目CD, CStr(ii))

                ' 空の配列でないか確認
                If IsArray(temp_abc) And Not IsEmptyArray(temp_abc) Then

                    car_sum = CountUniqueValues(temp_abc, 受入一覧.受入番号) '車両数
                    height_arr = col1D_from2DArray(temp_abc, 受入一覧.正味重量) '重量
                    height_sum = SumArray(height_arr)

                    For j = 3 To UBound(OS_Count, 2)
                        If OS_Count(1, j) = ii Then

                            ' OS_Count に代入 iがマイナスになるので注意
                            OS_Count(i - 1, j) = car_sum
                            OS_Count(i - 1, j + 1) = height_sum
                            Exit For             ' j を抜ける
                        End If
                    Next j
                End If
            Next ii

            culnum = culnum + 1
        Loop                                     'インプットデータ

        '        Next dim_num

        '出荷の履歴更新
        OS_ShipDist(i, 1) = 1
        OS_Count(i - 1, 1) = 1

next_loop:
    Next i


    '表を更新する
    With ws_Output_Cal.Range("A1").CurrentRegion
        .value = OS_ShipDist
    End With

    With wb_Output.Worksheets("カレンダー_台数").Range("A1").CurrentRegion
        .value = OS_Count
    End With


    'プログレスバー
    PB_nextStep = PB_nextStep + 3
    Call UpdateProgressBar(PB_currentStep, PB_nextStep, PB_totalSteps, progressBar)

    '==========テンプレート作成===========
    Dim loop_date As Long

    For loop_date = 4 To UBound(OS_ShipDist, 1)

        'カレンダー品目
        '履歴が１なら、テンプレート作成
        If OS_ShipDist(loop_date, 1) = 1 Then

            '===テンプレートのコピー=====================
            Dim sheet_name As String
            sheet_name = day_pickup(OS_ShipDist(loop_date, 2)) & "日" '日付毎にシート名を変更

            Dim ws_temp As Worksheet
            'newsheetのセット
            If IsSheetExists(wb_Output, sheet_name) = True Then
                Set ws_temp = wb_Output.Worksheets(sheet_name)
            Else
                'テンプレートから新しいワークシートを作成
                Set ws_temp = CreateNewWorksheet(ws_Output_Template, sheet_name)

                '日付を記入する
                With ws_temp
                    .Cells(2, "B") = OS_ShipDist(loop_date, 2)
                End With
            End If

            With ws_temp

                '=====テンプレートを埋める===========
                Dim fCell_abc As Range
                Set fCell_abc = .Cells(3, "E")
                Dim cul As Long

                '総合計
                fCell_abc.Offset(23, 4) = OS_ShipDist(loop_date, AS_item_cul.合計売上)
                fCell_abc.Offset(24, 4) = OS_ShipDist(loop_date, AS_item_cul.合計kg)

                '品目
                cul = 5
                For i = 0 To 5
                    For j = 0 To 2
                        fCell_abc.Offset(0 + i * 3, j) = OS_ShipDist(loop_date, cul)
                        fCell_abc.Offset(1 + i * 3, j) = OS_ShipDist(loop_date, cul + 1)
                        cul = cul + 2
                    Next j
                Next i

                '台数・重量
                cul = 3
                For i = 0 To 5
                    fCell_abc.Offset(i * 3 - 1, 5) = OS_Count(loop_date - 1, cul)
                    fCell_abc.Offset(i * 3 - 1, 6) = OS_Count(loop_date - 1, cul + 1)
                    cul = cul + 2
                Next i

            End With

ExitLoops:
            '履歴を2にする
            OS_ShipDist(loop_date, 1) = 2
            OS_Count(loop_date - 1, 1) = 2

        End If                                   '履歴が１なら
    Next loop_date

    '再度表を更新する
    With ws_Output_Cal.Range("A1").CurrentRegion
        .value = OS_ShipDist
    End With

    With wb_Output.Worksheets("カレンダー_台数").Range("A1").CurrentRegion
        .value = OS_Count
    End With

End Sub
