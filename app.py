import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import matplotlib.cm as cm

st.set_page_config(layout="wide", page_title="Sahte Görüntü Tespit Uygulaması")
st.title("🛡️ Sahte Görüntü Tespit Uygulaması")

# Görselleştirme Ayarları
def get_img_array(img_pil, size):
    img = img_pil.resize(size)
    array = keras_image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_occlusion_map(img_array, model, patch_size=40):
    img_array = img_array.copy()
    preds = model.predict(img_array, verbose=0)
    if isinstance(preds, dict):
        base_score = float(preds[list(preds.keys())[0]][0][0])
    elif isinstance(preds, list):
        base_score = float(preds[0][0]) if len(preds[0].shape) > 0 else float(preds[0])
    else:
        base_score = float(preds[0][0]) if preds.ndim > 1 else float(preds[0])
    
    h, w = img_array.shape[1:3]
    heatmap = np.zeros((h, w))
    
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            occluded = img_array.copy()
            occluded[0, y:y+patch_size, x:x+patch_size, :] = 0.5
            new_preds = model.predict(occluded, verbose=0)
            if isinstance(new_preds, dict):
                new_score = float(new_preds[list(new_preds.keys())[0]][0][0])
            elif isinstance(new_preds, list):
                new_score = float(new_preds[0][0]) if len(new_preds[0].shape) > 0 else float(new_preds[0])
            else:
                new_score = float(new_preds[0][0]) if new_preds.ndim > 1 else float(new_preds[0])
            heatmap[y:y+patch_size, x:x+patch_size] = np.abs(base_score - new_score)
    
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    return heatmap

def display_heatmap(img_pil, heatmap, alpha=0.5):
    img = keras_image.img_to_array(img_pil)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * alpha + img
    return keras_image.array_to_img(superimposed)

# Model Yükleme
@st.cache_resource
def load_custom_model():
    model_path = "trained_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı: {model_path}")
        return None
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata: {e}")
        return None

@st.cache_resource
def load_vit_model():
    model_yolu = "ViT_Model"
    if os.path.exists(model_yolu):
        try:
            return tf.keras.Sequential([tf.keras.layers.TFSMLayer(model_yolu, call_endpoint='serving_default')])
        except Exception as e:
            st.error(f"ViT yüklenemedi: {e}")
    return None


# Klasik Yöntemlerle Analiz
def klasik_yontem_analizi(image, algo):
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if algo == "ORB":
        detector = cv2.ORB_create(1000)
        norm_type = cv2.NORM_HAMMING
    elif algo == "AKAZE":
        detector = cv2.AKAZE_create()
        norm_type = cv2.NORM_HAMMING
    elif algo == "SURF":
        try:
            detector = cv2.xfeatures2d.SURF_create(400)
            norm_type = cv2.NORM_L2
        except:
            st.warning("⚠️ SURF desteklenmiyor, SIFT kullanılıyor.")
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
    else:
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2

    kp, des = detector.detectAndCompute(gray, None)
    if des is None or len(des) < 3:
        return img, 0

    matches = cv2.BFMatcher(norm_type).knnMatch(des, des, k=2)
    good = []
    for m, n in matches:
        if m.queryIdx != n.trainIdx:
            pt1 = kp[m.queryIdx].pt
            pt2 = kp[n.trainIdx].pt
            dist = np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
            if 50 < dist < 300 and m.distance < 0.5 * n.distance:
                good.append((pt1, pt2))

    sonuc = img.copy()
    for pt1, pt2 in good[:30]:
        cv2.line(sonuc, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255,0,0), 2)
    return sonuc, len(good)

# Yapay Zeka Tabanlı Analiz
def yapay_zeka_analizi(image, model_adi):
    sonuc_resmi = np.array(image.convert("RGB"))
    sahtecilik_orani = 0.0

    if model_adi == "Xception (CNN)":
        model = load_custom_model()
        if model is None:
            return sonuc_resmi, 0.0
        
        input_shape = model.input_shape[1:3]   
        img_array = get_img_array(image, input_shape)
        img_array = img_array / 255.0 
        preds = model.predict(img_array)

        if preds.shape[-1] == 1:
            sahtecilik_orani = (1.0 - float(preds[0][0])) * 100
        else:
            sahtecilik_orani = float(preds[0][1]) * 100
        
        with st.spinner("Görsel odak haritası oluşturuluyor..."):
            heatmap = make_occlusion_map(img_array, model, patch_size=50)
        sonuc_resmi = display_heatmap(image, heatmap)

    elif model_adi == "ViT (Vision Transformer)":
        model = load_vit_model()
        if model:
            img_array = get_img_array(image, (224, 224)) / 255.0
            preds = model.predict(img_array)
            if isinstance(preds, dict):
                ham = float(preds[list(preds.keys())[0]][0][0])
            else:
                ham = float(preds[0][0]) if preds.ndim > 1 else float(preds[0])
            sahtecilik_orani = (1 / (1 + np.exp(-ham))) * 100
            with st.spinner("ViT için bölge analizi yapılıyor..."):
                heatmap = make_occlusion_map(img_array, model, patch_size=30)
            sonuc_resmi = display_heatmap(image, heatmap)
    else:
        st.error("Bilinmeyen model")

    sahtecilik_orani = max(0.0, min(100.0, sahtecilik_orani))
    return sonuc_resmi, sahtecilik_orani

# Sayfa Düzeni
st.sidebar.header("⚙️ Analiz Ayarları")
algoritma_turu = st.sidebar.radio("Kategori:", ("Klasik Yöntemler", "Yapay Zeka Tabanlı Yöntemler"))
if algoritma_turu == "Klasik Yöntemler":
    secilen_algoritma = st.sidebar.selectbox("Algoritma:", ["ORB", "SIFT", "SURF", "AKAZE"])
else:
    secilen_algoritma = st.sidebar.selectbox("Model:", ["Xception (CNN)", "ViT (Vision Transformer)"])

uploaded_file = st.file_uploader("Lütfen bir resim dosyası seçin", type=['jpg', 'jpeg', 'png', 'gif'])

if uploaded_file:
    image = Image.open(uploaded_file)
    if st.button("🔍 Analizi Başlat", use_container_width=True):
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Orijinal Görüntü**")
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("**Analiz Sonucu**")
            
            if algoritma_turu == "Klasik Yöntemler":
                with st.spinner("Tarama yapıyor..."):
                    res, sayi = klasik_yontem_analizi(image, secilen_algoritma)
                
                st.image(res, caption=f"{secilen_algoritma} Tespit Çizgileri", use_container_width=True)
                st.metric("Tespit Edilen Kopya Alanı", sayi)
                
                if sayi > 50:
                    st.error("⚠️ ŞÜPHELİ")
                else:
                    st.success("✅ TEMİZ")
            else:
                with st.spinner("Tarama yapıyor..."):
                    res, oran = yapay_zeka_analizi(image, secilen_algoritma)
                
                st.image(res, caption="Modelin Odaklandığı Alanlar (Isı Haritası)", use_container_width=True)
                st.metric("Sahtecilik Olasılığı", f"%{oran:.2f}")
                st.progress(int(oran))
                
                if oran > 35:
                    st.error("🚨 MANİPÜLE EDİLMİŞ")
                else:
                    st.success("✅ ORİJİNAL")