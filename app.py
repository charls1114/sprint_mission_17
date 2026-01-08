import streamlit as st
from PIL import Image
from transformers import pipeline
from plotly import graph_objects as go

# 페이지 설정
st.set_page_config(page_title="이미지 분류", page_icon="🖼️")


# 앱 제목 및 설명
st.title("이미지 분류 앱 🖼️")
st.write("이미지를 업로드하면 사전 훈련된 모델을 사용하여 분류합니다.")
st.write("---")


# 모델 로드 함수
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")


# 이미지 분류 함수
def classify_image(_model, image):
    return _model(image)


model = load_model()
# 사이드바 - 이미지 업로드 및 카메라 입력
with st.sidebar:

    st.header("이미지 업로드")
    uploaded_images = st.file_uploader(
        "이미지 파일을 업로드하세요",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg"],
    )

    st.header("카메라 입력")
    cam_image = st.camera_input("카메라로 사진 찍기", key="camera_input")
    if cam_image is not None:
        if uploaded_images is None:
            uploaded_images = []
        uploaded_images.append(cam_image)

    # 세션 상태 초기화 / 이미지 추가하는 경우
    if st.session_state.get("uploaded_images") is None or len(
        st.session_state["uploaded_images"]
    ) < len(uploaded_images):
        st.session_state["uploaded_images"] = uploaded_images
    # 이미지 제거하는 경우
    else:
        for img in st.session_state["uploaded_images"]:
            if img not in uploaded_images:
                st.session_state[f"classified_{img.name}"] = False
                st.session_state["uploaded_images"].remove(img)

# 메인 영역 - 이미지 분류
if len(uploaded_images) > 0:
    for idx, uploaded_image in enumerate(uploaded_images):
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption=f"업로드된 이미지 {uploaded_image.name}")

        st.button(
            f"{uploaded_image.name} 분류 시작",
            # 여러 장의 이미지 업로드 시 각 버튼의 boolean 상태 유지를 위해 session_state 추가
            on_click=lambda i=idx: st.session_state.update(
                {f"classified_{uploaded_image.name}": True}
            ),
            key=f"classify_button_{uploaded_image.name}",
        )

        if st.session_state.get(f"classified_{uploaded_image.name}"):
            # 이미지 분류 및 결과 표시
            with st.spinner(f"{uploaded_image.name} 분류 중..."):
                results = classify_image(model, image)

            # 결과 표시
            if results:
                with st.expander(f"{uploaded_image.name} 분류 결과 보기"):
                    # 상위 K개 결과 선택 슬라이더
                    top_k = st.slider(
                        "표시할 예측 개수 선택",
                        min_value=1,
                        max_value=len(results),
                        value=5,
                        key=f"result_num_slider_{idx}",
                    )
                    # 상위 K개 결과 표시
                    for i, result in enumerate(results[:top_k]):
                        label = result["label"]
                        score = result["score"]
                        if i == 0:
                            st.write(f"**{label}**")
                        else:
                            st.write(label)
                        st.progress(score, text=f"{score*100:.1f}%")
                    # 막대 그래프 표시
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=[res["label"] for res in results[:top_k]],
                                y=[res["score"] * 100 for res in results[:top_k]],
                            )
                        ]
                    )
                    st.plotly_chart(fig)
else:
    st.info("왼쪽 사이드바에서 이미지를 업로드하세요.")
