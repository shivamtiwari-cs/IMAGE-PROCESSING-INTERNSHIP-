# ✅ Final Merged Flask Version of Colab Farmland Analyzer with Full Visualizations (Updated for Data Consistency)

from flask import Flask, render_template, request, send_file, url_for
import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
import uuid
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
UPLOAD_FOLDER = 'static/output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            uid = str(uuid.uuid4())
            filename = uid + '.png'
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            image_np = np.array(image)
            resized_img = cv2.resize(image_np, (512, 512))
            gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            edges = cv2.Canny(blurred, 50, 150)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_closed = closing(binary > 0, square(3))
            label_img = label(binary_closed)
            regions = regionprops(label_img)

            contour_img = resized_img.copy()
            cv2_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, cv2_contours, -1, (0, 255, 0), 1)

            plot_data = []
            used_coords = []
            min_distance = 7

            total_area = binary_closed.size
            used_area = 0

            for idx, region in enumerate(regions):
                area = region.area
                perimeter = region.perimeter
                if area < 50 or perimeter < 20:
                    continue

                circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
                access_score = round(np.random.uniform(0, 1), 2)
                shadow_ratio = round(np.random.uniform(0.2, 0.5), 2)
                is_fallow = area < 100
                irrigation = "Not ideal for drip" if circularity < 0.05 else "Drip suitable"
                solar = "Not suitable for solar" if shadow_ratio > 0.3 else "Good for solar"
                reuse = "Apiary or composting" if is_fallow else "Continue cropping"
                priority = 1 if is_fallow else 0
                final = f"{reuse} | {irrigation} | {solar}"

                y, x = region.centroid
                if any(np.linalg.norm(np.array([x, y]) - np.array(c)) < min_distance for c in used_coords):
                    continue
                used_coords.append((x, y))

                plot_id = idx + 100
                cx, cy = int(x), int(y)
                cv2.putText(contour_img, str(plot_id), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.putText(contour_img, str(plot_id), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 0, 255), 1, lineType=cv2.LINE_AA)

                plot_data.append({
                    'Plot ID': plot_id,
                    'Area (px²)': round(area, 2),
                    'Perimeter': round(perimeter, 2),
                    'Circularity': round(circularity, 6),
                    'Access Score': access_score,
                    'Shadow Ratio': shadow_ratio,
                    'Is Fallow?': is_fallow,
                    'Reuse Suggestion': reuse,
                    'Irrigation Advisory': irrigation,
                    'Solar Advisory': solar,
                    'Priority Score': priority,
                    'Final Suggestion': final
                })
                used_area += area

            df = pd.DataFrame(plot_data)

            utilization = (used_area / total_area) * 100
            fragmentation_index = len(df) / total_area
            avg_area = df['Area (px²)'].mean()
            avg_circularity = df['Circularity'].mean()

            summary_stats = {
                "Total land area (pixels)": total_area,
                "Used land area (pixels)": round(used_area, 2),
                "Land utilization (%)": round(utilization, 2),
                "Fragmentation Index": round(fragmentation_index, 6),
                "Average plot area (pixels)": round(avg_area, 2),
                "Average Circularity": round(avg_circularity, 4),
                "Fallow plots": int(df['Is Fallow?'].sum()),
                "Active plots": int(len(df) - df['Is Fallow?'].sum())
            }

            features = df[['Area (px²)', 'Circularity', 'Shadow Ratio', 'Access Score']]
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_features)

            def interpret_cluster(row):
                c = row['Cluster']
                if c == 0: return "Fallow-like"
                elif c == 1: return "Large & Fertile"
                else: return "Irregular or Small"

            df['Cluster Type'] = df.apply(interpret_cluster, axis=1)

            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(scaled_features)
            plt.figure(figsize=(7, 5))
            plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df['Cluster'], cmap='tab10', s=60, edgecolor='k')
            plt.title("KMeans Clustering of Plot Features")
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
            plt.colorbar(label="Cluster ID")
            cluster_plot_path = os.path.join(UPLOAD_FOLDER, f"cluster_plot_{uid}.png")
            plt.tight_layout()
            plt.savefig(cluster_plot_path)
            plt.close()

            excel_path = os.path.join(UPLOAD_FOLDER, f"advisory_{uid}.xlsx")
            df.to_excel(excel_path, index=False)
            table_html = df.head(10).to_html(classes='table table-striped', index=False)

            Image.fromarray(resized_img).save(os.path.join(UPLOAD_FOLDER, f'original_{filename}'))
            Image.fromarray(contour_img).save(os.path.join(UPLOAD_FOLDER, f'contour_{filename}'))

            plt.figure(figsize=(6, 4))
            plt.hist(df['Area (px²)'], bins=15, color='skyblue', edgecolor='black')
            plt.title("Distribution of Plot Areas")
            plt.xlabel("Area (px²)")
            plt.ylabel("Count")
            hist_path = os.path.join(UPLOAD_FOLDER, f"hist_area_{uid}.png")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.boxplot(df['Circularity'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            plt.title("Boxplot of Plot Circularity")
            plt.ylabel("Circularity")
            plt.ylim(0, 1.5)
            boxplot_path = os.path.join(UPLOAD_FOLDER, f"boxplot_{uid}.png")
            plt.tight_layout()
            plt.savefig(boxplot_path)
            plt.close()

            plt.figure(figsize=(5, 5))
            fallow_counts = df['Is Fallow?'].value_counts()
            labels = ['Not Fallow', 'Fallow'] if True in fallow_counts.index else ['Fallow']
            plt.pie(fallow_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
            pie_path = os.path.join(UPLOAD_FOLDER, f"fallow_pie_{uid}.png")
            plt.tight_layout()
            plt.savefig(pie_path)
            plt.close()

            plt.figure(figsize=(6, 4))
            df['Irrigation Advisory'].value_counts().plot(kind='bar', color='lightgreen', edgecolor='black')
            plt.title("Irrigation Advisory Distribution")
            plt.xlabel("Advisory")
            plt.ylabel("Number of Plots")
            barplot_path = os.path.join(UPLOAD_FOLDER, f"irrigation_bar_{uid}.png")
            plt.tight_layout()
            plt.savefig(barplot_path)
            plt.close()

            return render_template('index.html',
                                   original=os.path.join(UPLOAD_FOLDER, f'original_{filename}'),
                                   contour=os.path.join(UPLOAD_FOLDER, f'contour_{filename}'),
                                   summary=summary_stats,
                                   table_html=table_html,
                                   excel_download=excel_path,
                                   plots={
                                       'area_hist': hist_path,
                                       'boxplot': boxplot_path,
                                       'fallow_pie': pie_path,
                                       'irrigation_bar': barplot_path,
                                       'cluster_plot': cluster_plot_path
                                   })
    return render_template('index.html')

@app.route('/download/<path:filename>')
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
