kernel void label_with_id(global int *data, int width) {
  size_t x = get_global_id(0);
  size_t y = get_global_id(1);

  int loc = x * width + y;
  data[loc] = loc + 2;
}
