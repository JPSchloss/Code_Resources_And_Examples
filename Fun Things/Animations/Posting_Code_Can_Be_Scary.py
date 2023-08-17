import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from matplotlib.font_manager import FontProperties

def display_animation(texts, text_colors, bg_images, font_path, font_size=40, frame_duration=20):
    """
    A function to display an animated text sequence with different backgrounds.
    
    Parameters: 
        texts: List of texts to display
        text_colors: List of colors for each text
        bg_images: List of paths to background images
        font_path: Path to font file
        font_size: Font size for the texts
        frame_duration: Number of frames to display each text/image pair
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    # Load background images
    images = [mpimg.imread(img) for img in bg_images]
    current_image = ax.imshow(images[0], aspect='auto', extent=ax.get_xlim() + ax.get_ylim())

    font_properties = FontProperties(fname=font_path, size=font_size, weight='bold')
    text_obj = ax.text(0.5, 0.5, "", ha='center', va='center', 
                       fontproperties=font_properties, transform=ax.transAxes)

    def update(frame):
        idx = frame // frame_duration
        if idx < len(texts):
            current_image.set_array(images[idx])
            text_obj.set_text(texts[idx])
            text_obj.set_color(text_colors[idx])
        else:
            text_obj.set_text("")

    total_frames = frame_duration * len(texts)
    ani = FuncAnimation(fig, update, frames=total_frames, repeat=True)
    plt.show()

if __name__ == '__main__':
    display_animation(
        texts=["Posting Code\nCan Be Scary...", "But It's Good\nTo Share It Anyway!"],
        text_colors=['white', '#000000'],
        bg_images=['Fun Things/Animations/dark.jpg', 'Fun Things/Animations/chevron.jpg'],
        font_path="/System/Library/Fonts/Supplemental/AmericanTypewriter.ttc",
        font_size=40,
        frame_duration=20,
    )

    
