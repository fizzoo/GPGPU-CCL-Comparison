kernel void label_with_id(global int *data, unsigned int w) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int loc = w * y + x;
  if (data[loc] == 1) {
    data[loc] = loc + 2;
  }
}

kernel void neighbour_propagate(global int *data, unsigned int w, unsigned int h, global char *changed) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int oldlabel = data[w * y + x];
  int curlabel = oldlabel;
  int otherlabel = 0;

  if (curlabel == 0) {
    return;
  }

  if (y + 1 < h) {
    otherlabel = data[w * (y + 1) + (x)];
    if (otherlabel > curlabel) {
      curlabel = otherlabel;
    }
  }
  if (y - 1 < h) {
    otherlabel = data[w * (y - 1) + (x)];
    if (otherlabel > curlabel) {
      curlabel = otherlabel;
    }
  }
  if (x + 1 < w) {
    otherlabel = data[w * (y) + (x + 1)];
    if (otherlabel > curlabel) {
      curlabel = otherlabel;
    }
  }
  if (x - 1 < w) {
    otherlabel = data[w * (y) + (x - 1)];
    if (otherlabel > curlabel) {
      curlabel = otherlabel;
    }
  }

  if (curlabel < oldlabel) {
    *changed = 1;
    data[w * y + x] = curlabel;
  }
}

kernel void id_accessor(global int *data, int w) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int loc = w * y + x;
  data[loc] = data[loc];
}
