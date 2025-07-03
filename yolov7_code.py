# Mount Google Drive
drive.mount('/content/drive')

# Paths
video_folder = "/content/drive/My Drive/deeploycv/video"  # Folder containing videos
output_folder = "/content/drive/My Drive/deeploycv/output"  # Folder to save outputs

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# YOLOv8 model setup
model = YOLO("yolov8n.pt")
vehicle_classes = ["car", "truck", "bus", "motorbike"]

# Get list of video files
video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Process each video
for video_path in video_files:
    print(f"Processing video: {video_path}")
    video_name = os.path.basename(video_path).split('.')[0]

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        continue

    # Retrieve video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    video_duration = total_frames / fps  # Duration in seconds
    frame_skip = max(1, total_frames // 10)  # Extract 10 data points per video

    # Initialize variables
    frame_count = 0
    vehicle_counts = []
    timestamps = []

    # Process frames
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_skip == 0:  # Process selected frames
            results = model(frame, stream=True)
            frame_vehicle_count = 0
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0].item())
                    if model.names[cls] in vehicle_classes:
                        frame_vehicle_count += 1

            # Store results
            vehicle_counts.append(frame_vehicle_count)
            timestamps.append(frame_count / fps)  # Calculate timestamp

        frame_count += 1

    cap.release()

    # Save JSON data for this video
    with open(os.path.join(output_folder, f"{video_name}_vehicle_counts.json"), "w") as f:
        json.dump({"timestamps": timestamps, "vehicle_counts": vehicle_counts}, f)

    # Generate and save GASF matrix
    counts = np.array(vehicle_counts)
    normalized_counts = (counts - counts.min()) / (counts.max() - counts.min() + 1e-6)  # Avoid division by zero
    angles = np.arccos(normalized_counts)
    n = len(angles)
    gasf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gasf_matrix[i, j] = np.cos(angles[i] + angles[j])

    # Save GASF matrix as a NumPy array
    np.save(os.path.join(output_folder, f"{video_name}_gasf_matrix.npy"), gasf_matrix)

    # Save GASF matrix visualization (matrix only, no labels or axes)
    plt.figure(figsize=(6, 6))  # Set figure size
    plt.imshow(gasf_matrix, cmap="hot", interpolation="nearest")
    plt.axis("off")  # Turn off axes
    plt.tight_layout(pad=0)  # Remove padding around the image
    plt.savefig(os.path.join(output_folder, f"{video_name}_gasf_matrix.png"), bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Processing for {video_name} completed. Results saved to {output_folder}.")