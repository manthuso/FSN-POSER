import cv2
import mediapipe as mp

# --- Inicialização dos objetos do MediaPipe ---

# Inicializa a solução de detecção de pose (BlazePose)
mp_pose = mp.solutions.pose

# Inicializa a ferramenta para desenhar os pontos e conexões no corpo
mp_drawing = mp.solutions.drawing_utils

# Configura o detector de pose
# - static_image_mode=False: Para processar vídeo em tempo real.
# - model_complexity=1: Complexidade do modelo (0, 1 ou 2). 1 é um bom equilíbrio.
# - enable_segmentation=False: Não precisamos da máscara de segmentação do corpo.
# - min_detection_confidence=0.5: Confiança mínima para a detecção ser considerada válida.
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5)

# --- Processamento do Vídeo ---

# Inicia a captura de vídeo da webcam (0 é geralmente a webcam padrão)
cap = cv2.VideoCapture(0)

# Flag para garantir que os pontos 3D sejam impressos apenas uma vez
landmarks_printed = False

print("Pressione 'q' para sair.")

while cap.isOpened():
    # Lê um frame da webcam
    success, image = cap.read()
    if not success:
        print("Ignorando frame vazio da câmera.")
        continue

    # Para melhorar o desempenho, marcamos a imagem como não-gravável
    image.flags.writeable = False

    # O MediaPipe espera imagens em RGB, mas o OpenCV lê em BGR. Convertemos.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processa a imagem e detecta a pose
    results = pose.process(image_rgb)

    # Desmarca a imagem como não-gravável para poder desenhar nela
    image.flags.writeable = True

    # Converte a imagem de volta para BGR para exibição com o OpenCV
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # --- Extração e Impressão dos Pontos 3D ---

    # Verifica se os pontos 3D ("world landmarks") foram detectados
    if results.pose_world_landmarks and not landmarks_printed:
        print("--- Pontos do Corpo em Coordenadas 3D (em metros) ---")
        # Itera sobre cada ponto detectado
        for id, lm in enumerate(results.pose_world_landmarks.landmark):
            # Imprime o ID do ponto e suas coordenadas x, y, z
            print(f"Ponto ID {id} ({mp_pose.PoseLandmark(id).name}):")
            print(f"  x: {lm.x:.4f} m")
            print(f"  y: {lm.y:.4f} m")
            print(f"  z: {lm.z:.4f} m")
            # A visibilidade indica a probabilidade do ponto estar visível e não oculto
            print(f"  Visibilidade: {lm.visibility:.4f}")

        print("-" * 30)
        # Marcamos como impresso para não inundar o console.
        # Remova a linha abaixo se quiser impressão contínua.
        landmarks_printed = True

    # --- Desenho dos Pontos 2D na Tela ---

    # Desenha o esqueleto da pose (os pontos 2D) na imagem
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    # Mostra a imagem resultante em uma janela
    cv2.imshow('ugabuga ugabuga', image)

    # Interrompe o loop se a tecla 'q' for pressionada
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Liberação dos Recursos ---
pose.close()
cap.release()
cv2.destroyAllWindows()