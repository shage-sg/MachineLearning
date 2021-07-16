import cv2
import numpy as np
import matplotlib.pyplot as plt
import get_images
import local_enhancement
import os
import shutil
import split_leaves
import save_split_leaves
import cut_out_corrected_img
import show_images
import extract_vein_by_region_grow
import get_angle_vertical
from skimage import morphology
import get_top_and_bottom
import get_curvature
import save_in_csv_and_xlsx


# # Get Corrected_after Leaves Begin
# # clear "split_after" directory
# if os.path.isdir(r'./split_after'):
#     if os.listdir(r'./split_after'):
#         shutil.rmtree(r'./split_after')
#         os.mkdir(r'./split_after')

# # clear "corrected_after" directory
# if os.path.isdir(r'./corrected_after'):
#     if os.listdir(r'./corrected_after'):
#         shutil.rmtree(r'./corrected_after')
#         os.mkdir(r'./corrected_after')

# # get images
# leaf_split_before = get_images.get_images(r'./split_before')[0]
# leaves_split = split_leaves.split_leaves(leaf_split_before)
# save_split_leaves.save_split_leaves(leaves_split, leaf_split_before, r'./split_after')
# images = sprted(get_images.get_images(r'./split_after'))

# imgs_rotated = []
# imgs_shape = []

# for image in images:
#     print('Straightening {}'.format(image))
#     img_rotated_cut = cut_out_corrected_img.cut_out_corrected_img(image)
#     cv2.imwrite(r'./corrected_after/'+image.rpartition('/')[-1].rpartition('.')[-3][-1]+'.jpg', img_rotated_cut)
#     imgs_rotated.append(img_rotated_cut)
#     imgs_shape.append(img_rotated_cut.shape)

# column = 5
# img_joined = show_images.show_images(imgs_rotated, imgs_shape, column, alignment='left')
# img_ori = cv2.imread(leaf_split_before)
# # Get Corrected_after Leaves Begin

images = sorted(get_images.get_images(r'./corrected_after/'))

edges_canny = []
edges_equalized = []
edges_canny_shape = []
edges_equalized_shape = []

for image in images:
    img, img_equalized, edge_canny, edge_equalized = local_enhancement.local_enhancement(image)
    edges_canny.append(edge_canny)
    # edges_equalized.append(edge_equalized)
    edges_canny_shape.append(edge_canny.shape)
    # edges_equalized_shape.append(edge_equalized.shape)
    # plt.imshow(edge_canny, plt.cm.gray)
    # plt.show()

column = 5
edge_canny_joined = show_images.show_images(edges_canny, edges_canny_shape, column, alignment='left')
# edge_equalized_joined = show_images.show_images(edges_equalized, edges_equalized_shape, column, alignment='left')


# Extract main vein
main_veins = []
veins = []
main_veins_points = []
veins_points = []
main_veins_shape = []
veins_shape = []
veins_bgr = []
veins_bgr_shape = []
imgs = []
imgs_shape = []
tops = []
bottoms = []
curvatures = []
all_angles = []     # angles in all leaves
for i in range(len(edges_canny)):
    print('Extracting {}'.format(images[i]))
    vein, main_vein, vein_points, main_vein_points = \
        extract_vein_by_region_grow.extract_vein_by_region_grow(edges_canny[i], images[i], 150, (15, 15))

    # COLORIZATION BEGIN
    other_vein = cv2.subtract(vein, main_vein)
    _, contours, hierarchy = cv2.findContours(other_vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    small_perimeters = [j for j in contours if len(j) < 50]  # 删短周长的区域
    cv2.fillPoly(other_vein, small_perimeters, 0)
    other_vein_bgr = cv2.cvtColor(other_vein, cv2.COLOR_GRAY2BGR)
    _, contours, hierarchy = cv2.findContours(other_vein, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color_choice = [(0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255), (0, 64, 255)]
    masks = []
    individual_branchs = []
    num_more_color = 10
    while num_more_color >= 0:
        color_new = np.random.randint(0, 255, size=(1, 3))[0].tolist()
        if color_new not in color_choice:
            color_choice.append(color_new.copy())
            num_more_color -= 1
    for c in range(len(contours)-1, -1, -1):
        if cv2.arcLength(contours[c], True) < 100:
            contours.pop(c)
    # Append curvatures
    curvatures.append([])
    for c in range(len(contours)):
        if cv2.contourArea(contours[c]) < 60:
            continue
        canvas_t = np.zeros_like(vein)
        cv2.drawContours(canvas_t, contours, c, 1, cv2.FILLED)
        skn = morphology.skeletonize(canvas_t) * 255
        skn_axis = np.where(skn == 255)
        skn_axis = np.hstack(
            (skn_axis[0].reshape(skn_axis[0].shape[0], -1),
             skn_axis[1].reshape(skn_axis[1].shape[0], -1))
        )

        curvatures[-1].append(get_curvature.get_curvature(skn_axis[:, 1], skn_axis[:, 0]))

    for c in range(len(contours)):
        cv2.fillPoly(other_vein_bgr, [contours[c]], color_choice[c])
        t = np.zeros_like(other_vein_bgr, dtype=np.uint8)
        for j in range(other_vein_bgr.shape[0]):
            for k in range(other_vein_bgr.shape[1]):
                if other_vein_bgr[j][k].tolist() == list(color_choice[c]):
                    t[j][k] = [255, 255, 255]
        individual_branchs.append(t.copy())
    pts = []
    for c in range(len(contours)+1):
        pts.append([])
        if c == 0:
            skin = morphology.skeletonize((main_vein / 255).astype(np.uint8)) * 255
        else:
            gray = cv2.cvtColor(individual_branchs[c-1], cv2.COLOR_RGB2GRAY)
            ret, thr = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            skin = morphology.skeletonize((thr / 255).astype(np.uint8)) * 255
        for j in range(skin.shape[0]):
            for k in range(skin[i].shape[0]):
                if skin[j][k]:
                    pts[c].append([j, k])
    pts = np.array(pts)
    angles = []
    text_places = []
    for j in range(len(pts)):
        if j == 0:
            current_angle = get_angle_vertical.get_angle_vertical(pts[j])
        else:
            current_angle = get_angle_vertical.get_angle_vertical(pts[j], angles[0])
        # print('current_angle:', current_angle)
        angles.append(current_angle)
        text_places.append(pts[j][np.argmin(np.array(pts[j])[:, 0])])
    # print("text_places:", text_places)
    main_vein_bgr = cv2.cvtColor(main_vein, cv2.COLOR_GRAY2BGR)
    vein_bgr = cv2.add(main_vein_bgr, other_vein_bgr)
    img_t_avoid_covering = np.zeros_like(vein_bgr)
    axises_text = []
    for j in range(1, len(pts)):
        # cv2.circle(vein_bgr, (text_places[j][1], text_places[j][0]), 3, (161, 161), thickness=3)
        axis_text = [min(vein_bgr.shape[0]-25, max(0, text_places[j][1]-30)),
                     min(vein_bgr.shape[1], max(70, text_places[j][0]))]
        # the size of '8' is (70, 25), so the distance threshold can be 80
        for ax in axises_text:
            # Avoid overlapping
            if np.linalg.norm(np.asarray(ax) - np.asarray(axis_text)) < 70:
                if -25 < axis_text[0] - ax[0] < 0 and abs(axis_text[1] - ax[1]) < 70:
                    # Need moving left
                    axis_text[0] = max(0, axis_text[0] - (25 - abs(ax[0] - axis_text[0])))
                if 0 < axis_text[0] - ax[0] < 25 and abs(axis_text[1] - ax[1]) < 70:
                    # Need moving right
                    axis_text[0] = min(vein_bgr.shape[1]-25, axis_text[0] + (25 - abs(ax[0] - axis_text[0])))
                if -70 < axis_text[1] - ax[1] < 0 and abs(axis_text[0] - ax[0]) < 25:
                    # Need moving down
                    axis_text[1] = max(70, axis_text[1] + (70 - abs(ax[1] - axis_text[1])))
                if 0 < axis_text[1] - ax[1] < 70 and abs(axis_text[0] - ax[0]) < 25:
                    # Need moving up
                    axis_text[1] = min(vein_bgr.shape[0], axis_text[1] - (70 - abs(ax[1] - axis_text[1])))
        axises_text.append(axis_text)
        cv2.putText(vein_bgr, str(j),#+', '+str(round(angles[j], 1)),
                    tuple(axis_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color_choice[j-1], thickness=2)
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.imshow(vein_bgr, cmap="gray")
    # plt.title('Colored Veins with Angles')
    # plt.show()
    # COLORIZATION END

    # get top and bottom begin.
    img = cv2.imread(images[i])
    top, bottom = get_top_and_bottom.get_top_and_bottom(images[i], angles[0])
    tops.append(top)
    bottoms.append(bottom)
    cv2.circle(img, top[::-1], 10, (0, 0, 255), thickness=7)
    cv2.circle(img, bottom[::-1], 10, (0, 0, 255), thickness=7)
    imgs.append(img)
    imgs_shape.append(img.shape)
    # get top and bottom end
    veins.append(vein)
    veins_shape.append(vein.shape)
    veins_points.append(vein_points)
    main_veins.append(main_vein)
    main_veins_shape.append(main_vein.shape)
    main_veins_points.append(main_vein_points)
    veins_bgr.append(vein_bgr)
    veins_bgr_shape.append(vein_bgr.shape)
    all_angles.append(angles)

# collect vein_data
A4_name = get_images.get_images(r'./split_before')[0].split('/')[-1]
vein_data = 'vein_data'
if os.path.exists(vein_data):
    shutil.rmtree(vein_data)
os.mkdir(vein_data)
csv_file_general = os.path.join(vein_data, 'vein_general.csv')
if os.path.exists(csv_file_general):
    os.remove(csv_file_general)
save_in_csv_and_xlsx.save_in_csv_general(
    csv_file_general, A4_name, curvatures.copy(), all_angles
)
save_in_csv_and_xlsx.csv2xlsx(csv_file_general)
for i in range(len(images)):

    csv_file_curvature = os.path.join(vein_data, 'vein_curvatures_') + str(i) + '.csv'
    if os.path.exists(csv_file_curvature):
        os.remove(csv_file_curvature)
    for j in range(len(curvatures[i])):
        curvatures[i][j] = curvatures[i][j].tolist()
    save_in_csv_and_xlsx.save_in_csv_curvature(
        csv_file_curvature, A4_name, curvatures[i].copy()
    )    #, all_angles[i])
    save_in_csv_and_xlsx.csv2xlsx(csv_file_curvature)

column = 5
vein_joined = show_images.show_images(veins, veins_shape, column, alignment='left')
main_vein_joined = show_images.show_images(main_veins, main_veins_shape, column, alignment='left')
vein_bgr_joined = show_images.show_images(veins_bgr, veins_bgr_shape, column, alignment='left')
img_joined = show_images.show_images(imgs, imgs_shape, column, alignment='left')

# plt.shows
results = 'results'
if os.path.exists(results):
    shutil.rmtree(results)
os.mkdir(results)
fig_1, axes_1 = plt.subplots(1, 1, figsize=(16, 8))
axes_1.imshow(edge_canny_joined, cmap='gray')
axes_1.set_title('Cannied Edges')
plt.savefig(os.path.join(results, 'Cannied_Edges.jpg'), bbox_inches='tight')

fig_2, axes_2 = plt.subplots(1, 1, figsize=(16, 8))
axes_2.imshow(vein_joined, cmap='gray')
axes_2.set_title('Veins')
plt.savefig(os.path.join(results, 'Veins.jpg'), bbox_inches='tight')

fig_3, axes_3 = plt.subplots(1, 1, figsize=(16, 8))
axes_3.imshow(main_vein_joined, cmap='gray')
axes_3.set_title('Main Veins')
plt.savefig(os.path.join(results, 'Main_Veins.jpg'), bbox_inches='tight')

fig_4, axes_4 = plt.subplots(1, 1, figsize=(16, 8))
axes_4.imshow(cv2.cvtColor(vein_bgr_joined, cv2.COLOR_BGR2RGB))
axes_4.set_title('Colored Veins')
plt.savefig(os.path.join(results, 'Colored_Veins.jpg'), bbox_inches='tight')

fig_5, axes_5 = plt.subplots(1, 1, figsize=(16, 8))
axes_5.imshow(cv2.cvtColor(img_joined, cv2.COLOR_BGR2RGB))
axes_5.set_title('Leaves with tops and bottoms')
plt.savefig(os.path.join(results, 'Leaves_with_tops_and_bottoms.jpg'), bbox_inches='tight')

frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
frame.axes.get_xaxis().set_visible(False)
plt.show()
