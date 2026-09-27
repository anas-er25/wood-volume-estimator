import cv2
import numpy as np

def create_test_wood_image(filename, angle_name):
    """Crée une image de test avec du 'bois'"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Fond gris clair
    
    # Ajouter du texte
    cv2.putText(img, f'Test Image: {angle_name}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Dessiner des "rondins de bois" (rectangles bruns)
    for i in range(5):
        y = 150 + i * 50
        cv2.rectangle(img, (100, y), (550, y + 40), (70, 100, 170), -1)  # Brun en BGR
        cv2.rectangle(img, (100, y), (550, y + 40), (0, 0, 0), 2)  # Contour
    
    cv2.imwrite(filename, img)
    print(f"✅ {filename} créé")

# Générer 3 images
create_test_wood_image('static/test_photo_1.jpg', 'Vue Laterale')
create_test_wood_image('static/test_photo_2.jpg', 'Vue Frontale')
create_test_wood_image('static/test_photo_3.jpg', 'Vue Dessus')

print("\n📸 Images de test prêtes à l'emploi !")
print("Télécharge-les depuis l'interface web")