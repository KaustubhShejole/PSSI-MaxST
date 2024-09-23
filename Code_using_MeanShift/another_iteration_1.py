# if iteration_count_1 == 0:
#     file_to_show_path = seg_img_path+'_1_1.png'
# else:
#     file_to_show_path = seg_img_path+'_1_1_' + str(iteration_count_1) + '.png'
import numpy as np
from fun import get_cluster_at_point


file_to_show_path = seg_img_path+'_1_1_' + str(iteration_count_1) + '.png'
original_image = cv2.imread(file_to_show_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_rgb = np.copy(original_image)
image = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 3), dtype=np.uint8)

# Function to initialize variables

new_background_clicked_points = []
new_foreground_clicked_points = []
scribble_color = [0, 0, 255]


def init(image):
    global is_scribbling
    is_scribbling = False
    global points_list
    points_list = []
    global image_width
    global image_height
    image_width = image.shape[0]
    image_height = image.shape[1]
    global scribbling_length
    scribbling_length = scribbling_dimension

# Function to handle mouse press events


def on_press_foreground(event, clicked_points):
    global points_list
    global is_scribbling
    if event.xdata is not None and event.ydata is not None:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        is_scribbling = True
        # print('press' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
        points_list.append([x, y])
        # scribble1(x, y, clicked_points)
        cluster_label = get_cluster_at_point(x, y)
        clicked_points.append(cluster_label)
        # coords_superpixel = np.argwhere(labels_slic == cluster_label)
        # for (y, x) in coords_superpixel:
        #     image[y, x] = original_image[y, x]


def on_release(event, clicked_points):
    global is_scribbling
    is_scribbling = False
    global points_list
    scribble2(points_list, clicked_points)
    points_list = []
    plt.imshow(image)
    plt.draw()


def on_move(event, clicked_points):
    global points_list
    if is_scribbling:
        if event.xdata is not None and event.ydata is not None:
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            # print('move' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
            points_list.append([x, y])
            cluster_label = get_cluster_at_point(x, y)
            clicked_points.append(cluster_label)
            # scribble1(x, y, clicked_points)


def on_press_background(event, clicked_points):
    global points_list
    global is_scribbling
    if event.xdata is not None and event.ydata is not None:
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        is_scribbling = True
        # print('press' + str(x) + ' ' + str(y) + ' ' + str(clicked_points))
        points_list.append([x, y])
        # scribble1(x, y, clicked_points)
        cluster_label = get_cluster_at_point(x, y)
        clicked_points.append(cluster_label)

        # coords_superpixel = np.argwhere(labels_slic == cluster_label)
        # for (y, x) in coords_superpixel:
        #     image[y, x] = apply_translucent_scribble(
        #         image, x, y, [0, 0, 255], alpha=128)

# Function to perform scribbling (coloring)


def scribble2(points_list, clicked_points):
    for each_point in points_list:

        scribble3(each_point[0], each_point[1], clicked_points)


def scribble3(x, y, clicked_points):
    global image_width
    global image_height
    global scribbling_length
    pixels = []
    for i in range(-1*scribbling_length, scribbling_length+1):
        for j in range(-1*scribbling_length, scribbling_length+1):
            if (i != 0 or j != 0):
                if (((x+i) >= 0 and (x+i) < image_height) and ((y+j) >= 0 and (y+j) < image_width)):
                    pixels.append([x+i, y+j])
    for each_pixel in pixels:
        scribble(each_pixel[0], each_pixel[1], clicked_points)


def scribble1(x, y, clicked_points):
    pixels = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            if (i != 0 or j != 0):
                pixels.append([x+i, y+j])
    for each_pixel in pixels:
        scribble(each_pixel[0], each_pixel[1], clicked_points)
    plt.imshow(image)
    plt.draw()


def scribble(x, y, clicked_points):
    image[y, x] = scribble_color


# Initialize variables
init(image)

for i in range(image_rgb.shape[0]):
    for j in range(image_rgb.shape[1]):
        image[i, j] = image_rgb[i][j]

# Display the image and connect the mouse events
fig, ax = plt.subplots()
ax.set_title("Click foreground points")
ax.imshow(image)
ax.axis('off')
fig.canvas.mpl_connect('button_press_event', lambda event: on_press_foreground(
    event, new_foreground_clicked_points))
fig.canvas.mpl_connect('button_release_event', lambda event: on_release(
    event, new_foreground_clicked_points))
fig.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
    event, new_foreground_clicked_points))

plt.show()

# print("background")
scribble_color = [0, 255, 0]

fig1, ax1 = plt.subplots()
ax1.set_title("Click background points")
ax1.imshow(image)
ax1.axis('off')
fig1.canvas.mpl_connect('button_press_event',
                        lambda event: on_press_background(event, new_background_clicked_points))
fig1.canvas.mpl_connect('button_release_event', lambda event: on_release(
    event, new_background_clicked_points))
fig1.canvas.mpl_connect('motion_notify_event', lambda event: on_move(
    event, new_background_clicked_points))
plt.show()

plt.imshow(image)
plt.imsave(scribbled_img_path + '_1_1' +
           str(iteration_count_1 + 1)+'.png', image)
plt.axis('off')
plt.show()

# Start time
start_time = time.time()
for each_superpixel in new_background_clicked_points:
    if not G1.has_edge(each_superpixel, 'background'):
        G1.add_edge(each_superpixel, 'background', weight=np.inf)

for each_superpixel in new_foreground_clicked_points:
    if not G1.has_edge(each_superpixel, 'foreground'):
        G1.add_edge(each_superpixel, 'foreground', weight=np.inf)

maximum_spanning_tree = nx.maximum_spanning_tree(G1)


# helper(maximum_spanning_tree)
critical_edges = find_critical_edges(
    maximum_spanning_tree, 'background', 'foreground')
helper6(critical_edges, maximum_spanning_tree)
# End time
end_time = time.time()

# Calculate elapsed time
elapsed_time4 = end_time - start_time
print("Elapsed time for second iteration:", elapsed_time4, "seconds")
# helper1(image, maximum_spanning_tree, labels_slic)

# helper2(image, maximum_spanning_tree, labels_slic)
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_rgb = np.copy(original_image)
mask2 = generate_mask(image_rgb, maximum_spanning_tree, labels_slic)
plt.imshow(mask2)
plt.imsave(mask_img_path + '_1_1_'+str(iteration_count_1 + 1)+'.png',
           mask2)
plt.axis('off')
plt.show()
seg_img = helper3(image_rgb, maximum_spanning_tree, labels_slic)
# cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
plt.imshow(seg_img)
plt.imsave(seg_img_path + '_1_1_'+str(iteration_count_1 + 1)+'.png',
           seg_img)
plt.axis('off')
plt.show()
