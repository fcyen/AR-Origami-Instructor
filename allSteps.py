import cv2
import styles
from shapeDetection import findSquare, findTriangle, calculatedSquaredDistance
from shapeComparison import rotate
import draw

steps = []  # array of Step instances


class Step:
    def __init__(self, id, path, draw_fn, instruction, is_last_step=False, **kwargs):
        '''
        Parameters:
            path: str - path to reference image
            draw_fn: function - for drawing graphics, must accept bounding rectangle as parameter
            instruction: str - instruction to be displayed
        '''
        self.id = id
        self.is_last_step = is_last_step
        self.match_count = 0
        self.other_count = 0
        self.bouding_rect = []
        self.draw = draw_fn
        self.instruction = instruction
        self.kwargs = kwargs

        # -- get the reference image's contour --
        ref_img = cv2.imread(path)
        self.ref_bin = self.getBinarizedImage(ref_img)
        self.ref_cnt = self.detectContour(self.ref_bin, ref_img)
        self.vertices_count = len(self.ref_cnt)
        # cv2.imshow('Step '+str(id), ref_img)
        self.bounding_rect = cv2.minAreaRect(self.ref_cnt)
        if len(self.ref_cnt) < 1:
            print('No contour detected for reference image!')

    def detectContour(self, img, disp_img=[]):
        '''
        Returns first contour (approximated) detected from the binarized image given,
         empty array if nothing is detected

        Parameters:
            img - binarized image
            disp_img - image to draw detected contours on (for debugging)
        '''
        contours, _ = cv2.findContours(
            img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > 20000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.01*peri, True)
                # draw contour
                if len(disp_img) > 0:
                    cv2.drawContours(
                        disp_img, [approx], 0, (0, 0, 255), 2, offset=(0, 0))
                    cv2.putText(disp_img, str(
                        area), (approx[0][0][0]+50, approx[0][0][1]+50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('DetectContour', disp_img)

                # use the approximated contour
                result.append(approx)

        if len(result) > 0:
            return result[0]    # assume first one is the correct one
        else:
            return result

    def getBinarizedImage(self, img):
        ''' Binarized image using a threshold of 180 '''
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
            ret, img_bin = cv2.threshold(img_blur, 190, 255, cv2.THRESH_BINARY)
            return img_bin
        else:
            return img

    def compareShapes(self, cnt1, cnt2):
        ''' Returns True if both contours are similar '''
        d = -1
        if len(cnt1) > 0 and len(cnt2) > 0:
            d = cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I2, 0)

        if True:
            print('d = ' + str(d))
        # see excel file for experiment results
        if d < 0.6 and d >= 0:
            return True
        else:
            return False

    def checkShape(self, img_masked, img, debug=False):
        if "checkShapeOverride" in self.kwargs:
            # specifically for the first step, findSquare function is used
            if debug:
                shape = self.kwargs['checkShapeOverride'](img_masked, img)
            else:
                shape = self.kwargs['checkShapeOverride'](img_masked)

            if len(shape) > 0:
                shape_match = True
            else:
                shape_match = False

        else:
            if debug:
                cnt_feed = self.detectContour(img_masked, img)
            else:
                cnt_feed = self.detectContour(img_masked)

            # if len(cnt_feed) > 0:
            #     print('--')
            #     # cv2.drawContours(img_masked, [cnt_feed], 0, (0, 0, 255), 2)
            shape_match = self.compareShapes(self.ref_cnt, cnt_feed)

        # TODO: change back to 10
        if shape_match and self.match_count > 10:
            # get bounding rectangle
            # self.bounding_rect = cv2.minAreaRect(cnt_feed)
            # box = cv2.boxPoints(bounding_rect)
            # bounding_box = np.int0(box)
            self.match_count = 0
            self.other_count = 0
            return True

        elif shape_match:
            self.match_count += 1
        else:
            self.match_count = 0
            return False

    def perspectiveWarp(self, img, rect):
        # width = int(rect[1][0])
        # height = int(rect[1][1])
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # src_pts = box.astype("float32")
        # dst_pts = np.array([[0, height-1],
        #                     [0, 0],
        #                     [width-1, 0],
        #                     [width-1, height-1]], dtype="float32")
        # matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # warped = cv2.warpPerspective(img, matrix, (width, height))
        # return warped
        pass

    def alignContours(self, bimg):
        # cnt_feed = self.detectContour(bimg)
        # rect_feed = cv2.minAreaRect(cnt_feed)
        # warped_feed = self.perspectiveWarp(bimg, rect_feed)
        # warped_ref = self.perspectiveWarp(self.ref_bin, self.bounding_rect)
        # new_h, new_w = warped_feed.shape
        # resized_warped_ref = cv2.resize(warped_ref, (new_w, new_h))
        # plt.subplot(121), plt.imshow(warped_feed)
        # plt.subplot(122), plt.imshow(resized_warped_ref)
        # plt.show()
        pass

    def showNextStep(self, dimg, bimg):
        ''' Displays instruction graphics, returns True after a period of time to move on to next step '''
        if self.match_count < 50:
            if "checkShapeOverride" in self.kwargs:
                # specifically for the first step, findSquare function is used
                cnt_feed = self.kwargs['checkShapeOverride'](bimg, dimg)
            else:
                cnt_feed = self.detectContour(bimg)

            # make sure the number of vertices is correct
            if len(cnt_feed) == self.vertices_count:
                cv2.drawContours(dimg, [cnt_feed], 0, styles.GREEN, 3)
                self.draw(dimg, cnt_feed, [])
                self.match_count += 1
                cv2.putText(dimg, self.instruction, (100, 100),
                            cv2.FONT_HERSHEY_PLAIN, 3, styles.GREEN)
            else:
                print('Wrong number of vertices! Expected {} got {}'.format(
                    self.vertices_count, len(cnt_feed)))
                self.other_count += 1

                # if contour no longer matches, either
                # - break if instruction has been shown for sufficient duration
                # - reset count to zero so that instruction will be shown when contour matches again
                if self.other_count > 15:
                    if self.match_count > 10:
                        return True
                    else:
                        self.match_count = 0

            return False

        else:   # finish displaying instructions
            return True


def convertToIntPoint(point):
    return (int(point[0]), int(point[1]))


# ~~~~~~~~~~~~~~~~~ Step 1 ~~~~~~~~~~~~~~~~~~~
def draw1(img, ref_cnt, bounding_rect):
    pt1 = tuple(ref_cnt[0])
    pt2 = tuple(ref_cnt[1])
    pt3 = tuple(ref_cnt[2])
    pt4 = tuple(ref_cnt[3])
    # draws a line across the diagonal of the square
    cv2.line(img, pt2, pt4, (255, 0, 0), 3)
    draw.drawCurvedArrow(img, pt1, pt2, pt3, pt4)


instruction1 = "Fold paper in half"
# image path is unused because findSquare overrides the checkShape method
step1 = Step(1, 'assets/step2.png', draw1, instruction1,
             False, checkShapeOverride=findSquare)
step1.vertices_count = 4  # override vertices count
steps.append(step1)


# ~~~~~~~~~~~~~~~~~ Step 2a ~~~~~~~~~~~~~~~~~~~
def draw2a(img, ref_cnt, bounding_rect):
    # find distance between each vertex
    cnt = ref_cnt.reshape(3, 2)
    (ax, ay), (bx, by), (cx, cy) = cnt
    ab = (ax-bx)**2 + (ay-by)**2
    bc = (bx-cx)**2 + (by-cy)**2
    ac = (ax-cx)**2 + (ay-cy)**2

    long_edge = max(ab, bc, ac)
    if ab == long_edge:
        # mid = ((ax+bx)/2, (ay+by)/2)
        base1 = (ax, ay)
        base2 = (bx, by)
        top = (cx, cy)
    elif bc == long_edge:
        # mid = ((cx+bx)/2, (cy+by)/2)
        base1 = (bx, by)
        base2 = (cx, cy)
        top = (ax, ay)
    else:  # ac
        # mid = ((cx+ax)/2, (cy+ay)/2)
        base1 = (ax, ay)
        base2 = (cx, cy)
        top = (bx, by)

    # draw line
    line_start = base1
    r = 0.3827
    line_end = (r*base2[0] + (1-r)*top[0], r*base2[1] + (1-r)*top[1])
    line_end = convertToIntPoint(line_end)
    cv2.line(img, line_start, line_end, (255, 0, 0), 3)

    # draw curved arrow
    arrow_a = top
    arrow_b = ((base2[0]+top[0])/2, (base2[1]+top[1])/2)
    arrow_c = ((base2[0]+base1[0])/2, (base2[1]+base1[1])/2)
    arrow_d = ((base1[0]+top[0])/2, (base1[1]+top[1])/2)
    draw.drawCurvedArrow(img, arrow_a, arrow_b, arrow_c, arrow_d)


instruction2a = "Fold the top layer along the dotted line, aligning the edges."
step2a = Step(2, 'assets/step2.png', draw2a, instruction2a,
              False, checkShapeOverride=findTriangle)
steps.append(step2a)


# ~~~~~~~~~~~~~~~~~ Step 3a ~~~~~~~~~~~~~~~~~~~
def draw3a(img, ref_cnt, bounding_rect):
    # find distance between each vertex
    cnt = ref_cnt.reshape(3, 2)
    (ax, ay), (bx, by), (cx, cy) = cnt
    ab = (ax-bx)**2 + (ay-by)**2
    bc = (bx-cx)**2 + (by-cy)**2
    ac = (ax-cx)**2 + (ay-cy)**2

    long_edge = max(ab, bc, ac)
    if ab == long_edge:
        # mid = ((ax+bx)/2, (ay+by)/2)
        base1 = (ax, ay)
        base2 = (bx, by)
        top = (cx, cy)
    elif bc == long_edge:
        # mid = ((cx+bx)/2, (cy+by)/2)
        base1 = (bx, by)
        base2 = (cx, cy)
        top = (ax, ay)
    else:  # ac
        # mid = ((cx+ax)/2, (cy+ay)/2)
        base1 = (ax, ay)
        base2 = (cx, cy)
        top = (bx, by)

    # -- draw line
    r = 0.2
    line_start = (r*base1[0] + (1-r)*top[0], r*base1[1] + (1-r)*top[1])
    line_end = (r*base1[0] + (1-r)*base2[0], r*base1[1] + (1-r)*base2[1])
    line_start = convertToIntPoint(line_start)
    line_end = convertToIntPoint(line_end)
    cv2.line(img, line_start, line_end, (255, 0, 0), 3)

    # -- draw curved arrow
    # compute the center of the contour
    M = cv2.moments(ref_cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX, cY)
    arrow_a = rotate(base2, center, 15)
    arrow_b = rotate(top, center, 15)
    arrow_c = rotate(base1, center, 15)
    # find 4th vertex
    arrow_dX = arrow_a[0] + arrow_c[0] - arrow_b[0]
    arrow_dY = arrow_a[1] + arrow_c[1] - arrow_b[1]
    arrow_d = (arrow_dX, arrow_dY)
    draw.drawCurvedArrow(img, arrow_a, arrow_b, arrow_c, arrow_d)


instruction3a = "Fold all layers along the dotted line."
step3a = Step(3, 'assets/step2.png', draw3a, instruction3a,
              False, checkShapeOverride=findTriangle)
steps.append(step3a)


# ~~~~~~~~~~~~~~~~~ Step 4 ~~~~~~~~~~~~~~~~~~~
def draw4(img, ref_cnt, bounding_rect):
    pass


instruction4 = "Origami completed! Well done!!"
step4 = Step(4, 'assets/new_step4.png', draw4, instruction4, False)
steps.append(step4)


# ──────▄▀▄─────▄▀▄
# ─────▄█░░▀▀▀▀▀░░█▄
# ─▄▄──█░░░░░░░░░░░█──▄▄
# █▄▄█─█░░▀░░┬░░▀░░█─█▄▄█


# ~~~~~~~~~~~~~~~~~ Step 2 (deprecated) ~~~~~~~~~~~~~~~~~~~
def draw2(img, ref_cnt, bounding_rect):
    # find distance between each vertex
    cnt = ref_cnt.reshape(3, 2)
    (ax, ay), (bx, by), (cx, cy) = cnt
    ab = (ax-bx)**2 + (ay-by)**2
    bc = (bx-cx)**2 + (by-cy)**2
    ac = (ax-cx)**2 + (ay-cy)**2

    long_edge = max(ab, bc, ac)
    if ab == long_edge:
        # mid = ((ax+bx)/2, (ay+by)/2)
        base1 = (ax, ay)
        base2 = (bx, by)
        top = (cx, cy)
    elif bc == long_edge:
        # mid = ((cx+bx)/2, (cy+by)/2)
        base1 = (bx, by)
        base2 = (cx, cy)
        top = (ax, ay)
    else:  # ac
        # mid = ((cx+ax)/2, (cy+ay)/2)
        base1 = (ax, ay)
        base2 = (cx, cy)
        top = (bx, by)

    pt1 = (0.7*base1[0] + 0.3*top[0], 0.7*base1[1] + 0.3*top[1])
    pt2 = (0.7*base2[0] + 0.3*top[0], 0.7*base2[1] + 0.3*top[1])

    # find 4th point
    dx = base1[0] + base2[0] - top[0]
    dy = base1[1] + base2[1] - top[1]
    opp = (dx, dy)

    pt1 = convertToIntPoint(pt1)
    pt2 = convertToIntPoint(pt2)

    cv2.line(img, pt1, pt2, (255, 0, 0), 3)
    draw.drawCurvedArrow(img, top, base1, opp, base2)


instruction2 = "Fold both layers of paper to the right along the dotted line"
step2 = Step(2, 'assets/step2.png', draw2, instruction2, False)
# steps.append(step2)

# ~~~~~~~~~~~~~~~~~ Step 3 (deprecated) ~~~~~~~~~~~~~~~~~~~


def draw3(img, ref_cnt, bounding_rect, angle=0, scale_factor=1):
    ref_cnt = ref_cnt.reshape(7, 2)
    # identify correct points by finding the vertex connected to the longest edge
    vertex = 6
    edge_length = calculatedSquaredDistance(ref_cnt[0], ref_cnt[6])
    for i in range(6):
        l = calculatedSquaredDistance(ref_cnt[i], ref_cnt[i+1])
        if l > edge_length:
            edge_length = l
            vertex = i

    base1 = convertToIntPoint(ref_cnt[vertex])
    base2 = convertToIntPoint(ref_cnt[(vertex+1) % 7])
    top = convertToIntPoint(ref_cnt[(i+4) % 7])

    cv2.circle(img, base1, 8, styles.RED, -1)
    cv2.circle(img, base2, 8, styles.RED, -1)
    cv2.circle(img, top, 8, styles.RED)

    # paper lines
    cv2.line(img, base1, top, styles.GREEN, 3)
    cv2.line(img, base2, top, styles.GREEN, 3)

    # instruction lines
    pt1 = (0.2*top[0] + 0.8*base1[0], 0.2*top[1] + 0.8*base1[1])
    pt1 = convertToIntPoint(pt1)
    pt2 = (0.2*top[0] + 0.8*base2[0], 0.2*top[1] + 0.8*base2[1])
    pt2 = convertToIntPoint(pt2)
    cv2.line(img, pt1, pt2, (255, 0, 0), 3)


instruction3 = "Fold the top layer of paper to the left along the dotted line"
step3 = Step(3, 'assets/step3.png', draw3, instruction3, False)
# steps.append(step3)
