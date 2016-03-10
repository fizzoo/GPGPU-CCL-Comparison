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

kernel void gpu_label_mask_naive(global int *d, int w, int h) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    int loc = (y * w) + x;

    //west o north pos
    int checkW = (y * w) + (x - 1);
    int checkN = ((y - 1) * w) + x;

    //Init phase
    if (d[loc] == 1) {
        d[loc] = (loc + 2);
    } else {
        d[loc] = 0;
    }


    //Scan phase
    int scanTmp = d[loc]; 

//första koll om man kan minska labeln
    if (scanTmp != 0) {
        if (checkN >= 0 && checkN < w * h && d[checkN] > 0 
                   && d[checkN] < scanTmp) {
            scanTmp = d[checkN];
        }
        if (checkW >= 0 && checkW < w * h && d[checkW] > 0 
                   && d[checkW] < scanTmp) {
            scanTmp = d[checkW];
        }
    }
    d[loc] = scanTmp;
}

//är datarace kanske problemet? borde tänkt på det innan kanske
//finns väligt många index kontroller

kernel void gpu_label_mask_two (global int *d, int w, int h, global char *iter, global int *rLD) {
    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    int loc = (y * w) + x;

    if (d[loc] == loc + 2 && d[loc] > 0) {
        rLD[loc] = loc;
    } else {
        rLD[loc] = -1;
    }

    int lN = ((y - 1) * w) + x;
    int lE = (y * w) + (x + 1);
    int lS = ((y + 1) * w) + x;
    int lW = (y * w) + (x - 1);

    int size = w * h;


    //Analysis phase

    //man ska hitta en rot pixel, vilket är en pixel vars label är samma som 
    //dess init värde. här är då mitt fulhack jag skrev igår när jag 
    //var ledsen där man väntar på att att detta värde ska nå sig själv
    //istället för att leta upp det själv. antagligen det som går fel

    //tanken är att en rotpixel sprider sig från sig själv till pixlar omkring sig,
    //lite som cancer vilket var min inspiration 

    if (d[loc] > 0 && rLD[loc] != loc) {
        while(true) 
        {
            if (lN >= 0 && lN < size && rLD[lN] != -1 && d[lN] > 0) {
                rLD[loc] = rLD[lN]; 
                break;
            } else if (lE < size && rLD[lE] != -1 && d[lE] > 0) {
                rLD[loc] = rLD[lE];
                break;
            } else if (lS < size && rLD[lS] != -1 && d[lS] > 0) {
                rLD[loc] = rLD[lS];
                break;
            } else if (lW >= 0 && lW < size && rLD[lW] != -1 && d[lW] > 0) {
                rLD[loc] = rLD[lW];
                break;
            }
        }
    }


    //hittar man en rotpixel så tar man dess värde
    if (d[loc] > 0 && rLD[loc] != -1 && rLD[loc] >= 0 && rLD[loc] < size 
               && d[rLD[loc]] > 0) {
        d[loc] = d[rLD[loc]];
    }


    //Link phase
    //man kollar de två pixlarna till höger o vänster. Om de har en bättre
    //rotpixel så ändrar man sin egen rotpixels värde till något av deras

    int linkTmp = -1;

    if (d[loc] > 0 && rLD[loc] != -1 && rLD[loc] >= 0 && rLD[loc] < size 
               && d[rLD[loc]] > 0) {
        int linkTmp = rLD[loc];

        //i pdfen säger författarna att på cuda fanns det ett atomicMin
        //kommando som kunde undvika datarace för denna del, de säger inget om 
        //de andra dock

        if (lW >= 0 && lW < size && rLD[lW] != -1 && rLD[lW] >= 0 
               && rLD[lW] < size && d[lW] > 0 && d[linkTmp] < d[rLD[lW]]) {
            linkTmp = rLD[lW]; 
        }
        if (lE < size && rLD[lE] != -1 && rLD[lE] >= 0 && rLD[lE] < size
               && d[lE] > 0 && d[linkTmp] < d[rLD[lE]]) {
            linkTmp = rLD[lE];
        }

    }

    if (linkTmp != -1 && linkTmp >= 0 && linkTmp < size && rLD[loc] != -1
                && rLD[loc] >= 0 && rLD[loc] < size && d[rLD[loc]] > 0
                && d[linkTmp] > 0 && d[loc] > 0) {
        d[rLD[loc]] = d[linkTmp];
    }

    //Label phase
    //man tar de det uppdaterade värde från sin rotpixel

    if (d[loc] > 0 && rLD[loc] != -1 && rLD[loc] >= 0 && rLD[loc] < size && 
               d[rLD[loc]] > 0) {
        d[loc] = d[rLD[loc]];
    }

    //Rescan phase
    //gör om gör rätt

    if (d[loc] > 0) {
        if ((lN >= 0 && d[lN] > 0 && d[lN] != d[loc]) ||
            (lE < size && d[lE] > 0 && d[lE] != d[loc]) ||
            (lS < size && d[lS] > 0 && d[lS] != d[loc]) ||
            (lW >= 0 && d[lW] > 0 && d[lW] != d[loc])) {
            *iter = 1;
        }
    }
}
