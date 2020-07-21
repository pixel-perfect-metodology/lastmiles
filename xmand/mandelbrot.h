
/*
 * struct to pass params to dispatched thread
 * in generaly arrange the data from large to
 * small in terms of memory footprint.
 */
typedef struct {
    double r_translate, i_translate, magnify;
    double obs_x_width, obs_y_height;
    uint32_t (*v)[16][16][64][64];
    uint64_t ret_val;
    int vbox_x, vbox_y;
    int eff_width, eff_height;
    int vbox_w, vbox_h;
    uint32_t bail_out;
    int t_num;
    int t_total;
} thread_parm_t;

