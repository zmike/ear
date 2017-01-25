#include <Elementary.h>
#include <Emotion.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/calib3d/calib3d_c.h>

#define WEIGHT evas_object_size_hint_weight_set
#define ALIGN evas_object_size_hint_align_set
#define EXPAND(X) WEIGHT((X), EVAS_HINT_EXPAND, EVAS_HINT_EXPAND)
#define FILL(X) ALIGN((X), EVAS_HINT_FILL, EVAS_HINT_FILL)

static Evas_Object *win;
static Evas_Object *v;
static Evas_Object *rect;
static Eina_Array *rect_arr;
static IplImage *bgr, *ycrcb, *y, *cb, *cr, *cbh, *crh, *rsz, *gray;

static CvHaarClassifierCascade *haar;
static CvMemStorage *cvmem;
static Eina_Bool face = EINA_FALSE;

#define SIZE_LIMIT(w, h) (w > 300) || (h > 300)
#define SCALE 5

static void
_init(int w, int h)
{
   static Eina_Bool done = EINA_FALSE;

   if (done) return;
   bgr = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
   ycrcb = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3);
   y = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
   cb = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
   cr = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
   cbh = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);
   crh = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);
   if (SIZE_LIMIT(w, h))
     {
        rsz = cvCreateImage(cvSize(w/SCALE,h/SCALE), IPL_DEPTH_8U, 3);
        gray = cvCreateImage(cvSize(w/SCALE,h/SCALE), IPL_DEPTH_8U, 1);
     }
   else
     gray = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
   done = EINA_TRUE;
}

static inline int
_size_adj(float f, Eina_Bool scale, int w, int ww)
{
   if (scale)
     return lround(f * SCALE * ((double)ww / w));
   return lround(f * ((double)ww / w));
}

static void
_update_corners(CvPoint2D32f *corners, int w, int h)
{
   int rx, ry, rw, rh, wx, wy, ww, wh;
   Eina_Bool scale = SIZE_LIMIT(w, h);

   evas_object_geometry_get(elm_video_emotion_get(v), &wx, &wy, &ww, &wh);
   evas_object_show(rect);
   /* 176 x 132 */
   fprintf(stderr, "corners %ld,%ld %ldx%ld\n", lroundf(corners[0].x), lroundf(corners[0].y),
     lroundf(corners[19].x - corners[0].x),
     lroundf(corners[19].y - corners[0].y));
   rx = _size_adj(corners[0].x, scale, w, ww);
   ry = _size_adj(corners[0].y, scale, h, wh);
   evas_object_move(rect, wx + rx, wy + ry);
   rw = _size_adj(corners[19].x - corners[0].x, scale, w, ww);
   rh = _size_adj(corners[19].y - corners[0].y, scale, h, wh);
   evas_object_resize(rect, rw, rh);
   fprintf(stderr, "rect %d,%d %dx%d\n\n", rx, ry, rw, rh);
}

static void
_update_faces(CvSeq *faces, int w, int h)
{
   int i, wx, wy, ww, wh;
   int scale = (SIZE_LIMIT(w, h)) ? SCALE : 1;

   evas_object_geometry_get(elm_video_emotion_get(v), &wx, &wy, &ww, &wh);
   /* 176 x 132 */
   for (i = 0; i < faces->total && i < (int)eina_array_count(rect_arr); i++)
     {
        Eina_Rectangle *r = (void*)cvGetSeqElem(faces, i);
        Evas_Object *o = NULL;

        o = eina_array_data_get(rect_arr, i);

        evas_object_show(o);
        fprintf(stderr, "rect %d,%d %dx%d\n\n", r->x, r->y, r->w, r->h);
        evas_object_move(o, wx + (r->x * scale) * lround((double)ww / w), wy + (r->y * scale) * lround((double)wh / h));
        evas_object_resize(o, r->w * scale * lround((double)ww / w), r->h * scale * lround((double)wh / h));
     }
   for (; i < (int)eina_array_count(rect_arr); i++)
     evas_object_hide(eina_array_data_get(rect_arr, i));
}

static void
_frame(void *data EINA_UNUSED, Evas_Object *obj, void *event_info EINA_UNUSED)
{
   void **px;
   int w, h, corner_count, found;
   CvMat yuv, bbgr;
   double t0;
   CvSize search = cvSize(11,11);
   CvPoint2D32f corners[20];
   /* yuv referenced from https://github.com/mpenkov/opencv-yuv/blob/master/yuv.c */

   px = (void**)evas_object_image_data_get(emotion_object_image_get(obj), 1);
   if ((!px) || (!*px)) return;
   evas_object_image_size_get(emotion_object_image_get(obj), &w, &h);
   _init(w, h);

   if (SIZE_LIMIT(w, h))
     search = cvSize(4,4);

   //yuv = cvMat(h, w, CV_8UC2, *px);
   //bbgr = cvMat(h, w, CV_8UC3, NULL);
   t0 = ecore_time_unix_get();

   y->imageData = *px;
   cbh->imageData = *px;
   crh->imageData = *px;

   cvResize(cbh, cb, CV_INTER_CUBIC);
   //fprintf(stderr, "time2: %g\n", ecore_time_unix_get() - t0);
   t0 = ecore_time_unix_get();
   cvResize(crh, cr, CV_INTER_CUBIC);
   //fprintf(stderr, "time3: %g\n", ecore_time_unix_get() - t0);
   t0 = ecore_time_unix_get();
   cvMerge(y, cr, cb, NULL, ycrcb);
   //fprintf(stderr, "time4: %g\n", ecore_time_unix_get() - t0);
   t0 = ecore_time_unix_get();
   
   cvCvtColor(ycrcb, bgr, CV_YCrCb2BGR);
   //fprintf(stderr, "time5: %g\n", ecore_time_unix_get() - t0);
   t0 = ecore_time_unix_get();
   //cvCvtColor(&yuv, &bbgr, CV_YUV2BGR_I420);
   if (SIZE_LIMIT(w, h))
     cvResize(bgr, rsz, CV_INTER_CUBIC);
   else
     rsz = bgr;

   cvCvtColor(rsz, gray, CV_BGR2GRAY);
   if (face)
     {
        CvSeq *faces = cvHaarDetectObjects(gray, haar, cvmem, 1.1, 3, CV_HAAR_SCALE_IMAGE, cvSize(30, 30), cvSize(0, 0));

        if (faces)
          _update_faces(faces, w, h);
        else
          evas_object_hide(rect);
     }
   else
     {
        found = cvFindChessboardCorners(rsz, cvSize(5, 4), corners, &corner_count,
          CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE | CV_CALIB_CB_FAST_CHECK);

        cvFindCornerSubPix(gray, corners, corner_count, search, cvSize(-1,-1),
          cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1 ));
        if (found)
          _update_corners(corners, w, h);
        else
          evas_object_hide(rect);
     }
}

static void
_fini(void *data EINA_UNUSED, Evas_Object *obj EINA_UNUSED, void *event_info EINA_UNUSED)
{
   evas_object_del(win);
}

static void
_start(void *data EINA_UNUSED, Evas_Object *obj, void *event_info EINA_UNUSED)
{
   int w, h;
   emotion_object_size_get(obj, &w, &h);
   evas_object_resize(win, w, h);
}

static Eina_Bool
_key(void *d EINA_UNUSED, int t EINA_UNUSED, Ecore_Event_Key *ev)
{
   if (eina_streq(ev->key, "space"))
     {
        if (elm_video_is_playing_get(v))
          elm_video_pause(v);
        else
          elm_video_play(v);
     }
   else if (eina_streq(ev->key, "q"))
     exit(0);
   return ECORE_CALLBACK_RENEW;
}

int
main(int argc, char *argv[])
{
   CvMemStorage *haarmem = cvCreateMemStorage(0);

   rect_arr = eina_array_new(1);
   haar = cvLoad("haarcascade_frontalface_default.xml", haarmem, NULL, NULL);
   cvmem = cvCreateMemStorage(0);
   elm_init(argc, argv);
   elm_policy_set(ELM_POLICY_QUIT, ELM_POLICY_QUIT_LAST_WINDOW_CLOSED);
   win = elm_win_util_standard_add("ear", "ear");
   elm_win_autodel_set(win, 1);
   v = elm_video_add(win);
   rect = evas_object_rectangle_add(evas_object_evas_get(win));
   evas_object_color_set(rect, 255, 0, 0, 255);
   evas_object_show(rect);
   FILL(v);
   elm_win_resize_object_add(win, v);
   if (argc != 2)
     {
        if (argc == 3)
          {
             if (eina_streq(argv[2], "face"))
               face = 1;
             else
               return 1;
          }
        else
          return 1;
     }
   while (eina_array_count(rect_arr) < 30)
     {
        Evas_Object *o = evas_object_rectangle_add(evas_object_evas_get(win));
        evas_object_color_set(o, 255 * (double)150/255, 0, 0, 150);
        eina_array_push(rect_arr, o);
     }
   elm_video_file_set(v, argv[1]);
   elm_video_play(v);
   evas_object_smart_callback_add(elm_video_emotion_get(v), "frame_decode", _frame, NULL);
   evas_object_smart_callback_add(elm_video_emotion_get(v), "playback_started", _start, NULL);
   evas_object_smart_callback_add(elm_video_emotion_get(v), "playback_finished", _fini, NULL);
   ecore_event_handler_add(ECORE_EVENT_KEY_DOWN, _key, NULL);
   evas_object_show(v);
   evas_object_show(win);
   elm_run();
   return 0;
}
