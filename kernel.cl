kernel void label_with_id(global int *data, unsigned int w) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int loc = w * y + x;
  if (data[loc] == 1) {
    data[loc] = loc + 2;
  }
}

kernel void neighbour_propagate(global int *data, unsigned int w,
                                unsigned int h, global char *changed) {
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
    if (otherlabel && otherlabel < curlabel) {
      curlabel = otherlabel;
    }
  }
  if (y - 1 < h) {
    otherlabel = data[w * (y - 1) + (x)];
    if (otherlabel && otherlabel < curlabel) {
      curlabel = otherlabel;
    }
  }
  if (x + 1 < w) {
    otherlabel = data[w * (y) + (x + 1)];
    if (otherlabel && otherlabel < curlabel) {
      curlabel = otherlabel;
    }
  }
  if (x - 1 < w) {
    otherlabel = data[w * (y) + (x - 1)];
    if (otherlabel && otherlabel < curlabel) {
      curlabel = otherlabel;
    }
  }

  if (curlabel < oldlabel) {
    *changed = 1;
    data[w * y + x] = curlabel;
  }
}

kernel void plus_propagate(global int *data, unsigned int w, unsigned int h,
                           global char *changed) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int oldlabel = data[w * y + x];
  int curlabel = oldlabel;
  int otherlabel = 0;
  int diff;

  if (curlabel == 0) {
    return;
  }

  diff = 1;
  while (true) {
    if (y + diff >= h) {
      break;
    }
    otherlabel = data[w * (y + diff) + (x)];
    if (otherlabel == 0) {
      break;
    }
    if (otherlabel < curlabel) {
      curlabel = otherlabel;
    }
    ++diff;
  }
  diff = 1;
  while (true) {
    if (y - diff >= h) {
      break;
    }
    otherlabel = data[w * (y - diff) + (x)];
    if (otherlabel == 0) {
      break;
    }
    if (otherlabel < curlabel) {
      curlabel = otherlabel;
    }
    ++diff;
  }
  diff = 1;
  while (true) {
    if (x + diff >= w) {
      break;
    }
    otherlabel = data[w * y + (x + diff)];
    if (otherlabel == 0) {
      break;
    }
    if (otherlabel < curlabel) {
      curlabel = otherlabel;
    }
    ++diff;
  }
  diff = 1;
  while (true) {
    if (x - diff >= w) {
      break;
    }
    otherlabel = data[w * y + (x - diff)];
    if (otherlabel == 0) {
      break;
    }
    if (otherlabel < curlabel) {
      curlabel = otherlabel;
    }
    ++diff;
  }

  if (curlabel < oldlabel) {
    *changed = 1;
    data[w * y + x] = curlabel;
  }
}

kernel void lineedit_right(global int *data, int w, int h,
                           global char *changed) {
  int x = 0;
  int y = get_global_id(0);
  int lowest = 1 << 30;

  while (x < w) {
    int curlabel = data[w * y + x];

    if (curlabel == 0) {
      lowest = 1 << 30;
    } else {
      if (curlabel < lowest) {
        lowest = curlabel;
      } else if (curlabel > lowest) {
        data[w * y + x] = lowest;
        *changed = 1;
      }
    }

    ++x;
  }
}

kernel void lineedit_left(global int *data, int w, int h,
                          global char *changed) {
  int x = w - 1;
  int y = get_global_id(0);
  int lowest = 1 << 30;

  while (x >= 0) {
    int curlabel = data[w * y + x];

    if (curlabel == 0) {
      lowest = 1 << 30;
    } else {
      if (curlabel < lowest) {
        lowest = curlabel;
      } else if (curlabel > lowest) {
        data[w * y + x] = lowest;
        *changed = 1;
      }
    }

    --x;
  }
}

kernel void lineedit_up(global int *data, int w, int h, global char *changed) {
  int x = get_global_id(0);
  int y = 0;
  int lowest = 1 << 30;

  while (y < h) {
    int curlabel = data[w * y + x];

    if (curlabel == 0) {
      lowest = 1 << 30;
    } else {
      if (curlabel < lowest) {
        lowest = curlabel;
      } else if (curlabel > lowest) {
        data[w * y + x] = lowest;
        *changed = 1;
      }
    }

    ++y;
  }
}

kernel void lineedit_down(global int *data, int w, int h,
                          global char *changed) {
  int x = get_global_id(0);
  int y = h - 1;
  int lowest = 1 << 30;

  while (y >= 0) {
    int curlabel = data[w * y + x];

    if (curlabel == 0) {
      lowest = 1 << 30;
    } else {
      if (curlabel < lowest) {
        lowest = curlabel;
      } else if (curlabel > lowest) {
        data[w * y + x] = lowest;
        *changed = 1;
      }
    }

    --y;
  }
}

kernel void id_accessor(global int *data, int w) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int loc = w * y + x;
  data[loc] = data[loc];
}
