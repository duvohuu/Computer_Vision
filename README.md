# CODE
- Exercise → Exercise_n → Question_n → Code + Video 
# BÁO CÁO
- Chapter → Section → Subsection → Subsubsection → Dot → '-' → '+'

## Cấu trúc thư mục ảnh
Tất cả ảnh đặt trong thư mục chính `pictures/`. Mỗi chương có một thư mục con chứa ảnh của chương đó.

Cấu trúc:
- pictures/
  - chapter_1/
  - chapter_2/
  - chapter_3/
  - ...

Quy ước đặt tên tệp ảnh:
- Định dạng: cx_py_abc.ext
  - c: ký tự 'c' (chỉ chapter)
  - x: số chương (ví dụ 1, 2, 10)
  - _p: phân cách và chữ 'p' (picture)
  - y: số thứ tự ảnh trong chương (ví dụ 01, 02; khuyến khích 2 chữ số)
  - _abc: mô tả ngắn về nội dung (không dấu, dùng underscore để cách)
  - .ext: phần mở rộng (png, jpg, pdf, ...)

Ví dụ:
- c1_p01_overview.png      (Ảnh 1 của chương 1, mô tả: overview)
- c2_p03_topography.jpg    (Ảnh 3 của chương 2, mô tả: topography)
- c10_p12_cross_section.pdf

## Template cho hình ảnh:
```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=1\textwidth]{pictures/image.png}
    \caption{Mô tả hình ảnh}
    \label{fig:label}
\end{figure}
```

## Template cho bảng:
```latex
\begin{table}[H]
    \centering
    \begin{tabular}{|c|c|c|}
        \hline
        \textbf{Cột 1} & \textbf{Cột 2} & \textbf{Cột 3} \\
        \hline
        Dữ liệu 1 & Dữ liệu 2 & Dữ liệu 3 \\
        \hline
    \end{tabular}
    \caption{Mô tả bảng}
    \label{tab:label}
\end{table}
```