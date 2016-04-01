kernel void label_with_id(global int *data, int w, int h) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= w || y >= h) {
    return;
  }

  int loc = w * y + x;
  if (data[loc] == 1) {
    data[loc] = loc + 2;
  }
}

kernel void neighbour_propagate(global int *data, int w, int h,
                                global char *changed) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= w || y >= h) {
    return;
  }

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
  if (y - 1 >= 0) {
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
  if (x - 1 >= 0) {
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

kernel void plus_propagate(global int *data, int w, int h,
                           global char *changed) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= w || y >= h) {
    return;
  }

  int oldlabel = data[w * y + x];
  int curlabel = oldlabel;
  int otherlabel = 0;
  int diff;

  if (curlabel == 0) {
    return;
  }

  diff = 1;
  while (true) {
    if (y + diff < 0 || y + diff >= h) {
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
    if (y - diff < 0 || y - diff >= h) {
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
    if (x + diff < 0 || x + diff >= w) {
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
    if (x - diff < 0 || x - diff >= w) {
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

int find_set(global int *data, int loc) {
  // All loc of found elements should be in range.
  // Also assuming there are no cycles in the links.
  while (loc != data[loc] - 2) {
    loc = data[loc] - 2;
  }

  // +2 is the correct LABEL of root at LOCATION loc
  return loc + 2;
}

kernel void union_find(global int *data, int w, int h, global char *changed) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  if (x >= w || y >= h) {
    return;
  }

  int oldlabel = data[w * y + x];
  int lowest = oldlabel;

  if (oldlabel == 0) {
    return;
  }

  bool ok_N = y - 1 >= 0 && data[w * (y - 1) + (x)];
  bool ok_E = x + 1 < w && data[w * (y) + (x + 1)];
  bool ok_S = y + 1 < h && data[w * (y + 1) + (x)];
  bool ok_W = x - 1 >= 0 && data[w * (y) + (x - 1)];
  int root_N;
  int root_E;
  int root_S;
  int root_W;

  if (ok_N) {
    root_N = find_set(data, w * (y - 1) + (x));
    if (root_N < lowest) {
      lowest = root_N;
    }
  }
  if (ok_E) {
    root_E = find_set(data, w * (y) + (x + 1));
    if (root_E < lowest) {
      lowest = root_E;
    }
  }
  if (ok_S) {
    root_S = find_set(data, w * (y + 1) + (x));
    if (root_S < lowest) {
      lowest = root_S;
    }
  }
  if (ok_W) {
    root_W = find_set(data, w * (y) + (x - 1));
    if (root_W < lowest) {
      lowest = root_W;
    }
  }

  if (lowest < oldlabel) {
    *changed = 1;
    data[w * y + x] = lowest;
    if (ok_N && root_N > lowest) {
      data[root_N - 2] = lowest;
    }
    if (ok_E && root_E > lowest) {
      data[root_E - 2] = lowest;
    }
    if (ok_S && root_S > lowest) {
      data[root_S - 2] = lowest;
    }
    if (ok_W && root_W > lowest) {
      data[root_W - 2] = lowest;
    }
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

kernel void lines_up(global int *data, int w, int h, global char *changed) {
  int x = get_global_id(0);
  int y = 0;
  char localchanged = 0;

  while (y < h) {
    if (!data[w * y + x]) {
      ++y;
      continue;
    }
    int i = y;
    int min = data[w * y + x];
    int tmpmin = min;

    while (i < h && (tmpmin = data[w * i + x])) {
      if (tmpmin != min) {
        localchanged = 1; // Not all equal, i.e. will have change
      }
      if (tmpmin < min) {
        min = tmpmin;
      }
      ++i;
    }

    while (y != i) {
      data[w * y + x] = min;
      ++y;
    }
  }

  if (localchanged) {
    *changed = localchanged;
  }
}

kernel void lines_right(global int *data, int w, int h, global char *changed) {
  int x = 0;
  int y = get_global_id(0);
  char localchanged = 0;

  while (x < w) {
    if (!data[w * y + x]) {
      ++x;
      continue;
    }
    int i = x;
    int min = data[w * y + x];
    int tmpmin = min;

    while (i < w && (tmpmin = data[w * y + i])) {
      if (tmpmin != min) {
        localchanged = 1; // Not all equal, i.e. will have change
      }
      if (tmpmin < min) {
        min = tmpmin;
      }
      ++i;
    }

    while (x != i) {
      data[w * y + x] = min;
      ++x;
    }
  }

  if (localchanged) {
    *changed = localchanged;
  }
}

kernel void id_accessor(global int *data, int w) {
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);

  int loc = w * y + x;
  data[loc] = data[loc];
}

// Can't allocate if not known at compile-time anyway
#define lw 32
#define lh 8

kernel void solve_locally_nprop(global int *data, int w, int h) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int x = get_global_id(0);
  int y = get_global_id(1);

  char valid = 1;
  local int buffer[lw * lh];
  local char changed;

  if (y >= h || x >= w) {
    valid = 0;
  }
  buffer[lw * ly + lx] = valid ? data[w * y + x] : 0;
  changed = 1;

  while (changed) {
    barrier(CLK_LOCAL_MEM_FENCE);
    changed = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) {
      int min = 1 << 30;
      int tmp;
      if (lx > 0) {
        tmp = buffer[lw * (ly) + (lx - 1)];
        if (tmp && tmp < min) {
          min = tmp;
        }
      }
      if (lx < lw - 1) {
        tmp = buffer[lw * (ly) + (lx + 1)];
        if (tmp && tmp < min) {
          min = tmp;
        }
      }
      if (ly > 0) {
        tmp = buffer[lw * (ly - 1) + (lx)];
        if (tmp && tmp < min) {
          min = tmp;
        }
      }
      if (ly < lh - 1) {
        tmp = buffer[lw * (ly + 1) + (lx)];
        if (tmp && tmp < min) {
          min = tmp;
        }
      }
      if (min < buffer[lw * ly + lx]) {
        changed = 1;
        buffer[lw * ly + lx] = min;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (valid) {
    data[w * y + x] = buffer[lw * ly + lx];
  }
}

kernel void solve_locally_plus(global int *data, int w, int h) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int x = get_global_id(0);
  int y = get_global_id(1);

  char valid = 1;
  int diff;
  local int buffer[lw * lh];
  local char changed;

  if (y >= h || x >= w) {
    valid = 0;
  }
  buffer[lw * ly + lx] = valid ? data[w * y + x] : 0;
  changed = 1;

  while (changed) {
    barrier(CLK_LOCAL_MEM_FENCE);
    changed = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (valid) {
      int min = 1 << 30;
      int tmp;

      diff = 1;
      while (lx - diff >= 0) {
        tmp = buffer[lw * (ly) + (lx - diff)];
        if (!tmp) {
          break;
        }
        if (tmp < min) {
          min = tmp;
        }
        ++diff;
      }

      diff = 1;
      while (lx + diff < lw) {
        tmp = buffer[lw * (ly) + (lx + diff)];
        if (!tmp) {
          break;
        }
        if (tmp < min) {
          min = tmp;
        }
        ++diff;
      }

      diff = 1;
      while (ly - diff >= 0) {
        tmp = buffer[lw * (ly - diff) + (lx)];
        if (!tmp) {
          break;
        }
        if (tmp < min) {
          min = tmp;
        }
        ++diff;
      }

      diff = 1;
      while (ly + diff < lh) {
        tmp = buffer[lw * (ly + diff) + (lx)];
        if (!tmp) {
          break;
        }
        if (tmp < min) {
          min = tmp;
        }
        ++diff;
      }
      if (min < buffer[lw * ly + lx]) {
        changed = 1;
        buffer[lw * ly + lx] = min;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (valid) {
    data[w * y + x] = buffer[lw * ly + lx];
  }
}

kernel void plus_once_locally(global int *data, int w, int h) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int x = get_global_id(0);
  int y = get_global_id(1);

  int diff;
  local int buffer[lw * lh];

  if (y >= h || x >= w) {
    buffer[lw * ly + lx] = 0;
    return;
  }
  buffer[lw * ly + lx] = data[w * y + x];
  if (buffer[lw * ly + lx] == 0) {
    return;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int min = 1 << 30;
  int tmp;

  diff = 1;
  while (lx - diff >= 0) {
    tmp = buffer[lw * (ly) + (lx - diff)];
    if (!tmp) {
      break;
    }
    if (tmp < min) {
      min = tmp;
    }
    ++diff;
  }

  diff = 1;
  while (lx + diff < lw) {
    tmp = buffer[lw * (ly) + (lx + diff)];
    if (!tmp) {
      break;
    }
    if (tmp < min) {
      min = tmp;
    }
    ++diff;
  }

  diff = 1;
  while (ly - diff >= 0) {
    tmp = buffer[lw * (ly - diff) + (lx)];
    if (!tmp) {
      break;
    }
    if (tmp < min) {
      min = tmp;
    }
    ++diff;
  }

  diff = 1;
  while (ly + diff < lh) {
    tmp = buffer[lw * (ly + diff) + (lx)];
    if (!tmp) {
      break;
    }
    if (tmp < min) {
      min = tmp;
    }
    ++diff;
  }
  if (min < buffer[lw * ly + lx]) {
    buffer[lw * ly + lx] = min;
    data[w * y + x] = min;
  }
}
