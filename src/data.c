#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "dubnet.h"
#include "jcr.h"
#include "image.h"

tensor image_to_tensor(image im)
{
    tensor t = tensor_vmake(3, im.c, im.h, im.w);
    memcpy(t.data, im.data, tensor_len(t)*sizeof(float));
    return t;
}

data random_batch(data d, int n)
{
    size_t *sx = calloc(d.x.n, sizeof(size_t));
    memcpy(sx, d.x.size, d.x.n*sizeof(size_t));
    sx[0] = n;

    size_t *sy = calloc(d.y.n, sizeof(size_t));
    memcpy(sy, d.y.size, d.y.n*sizeof(size_t));
    sy[0] = n;

    tensor x = tensor_make(d.x.n, sx);
    tensor y = tensor_make(d.y.n, sy);
    size_t i;
    for(i = 0; i < n; ++i){
        size_t ind = rand()%d.x.size[0];
        tensor_axpy_(1, tensor_get_(d.x, ind), tensor_get_(x, i));
        tensor_axpy_(1, tensor_get_(d.y, ind), tensor_get_(y, i));
    }
    data c;
    c.x = x;
    c.y = y;
    free(sx);
    free(sy);
    return c;
}

list *get_lines(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) {
        fprintf(stderr, "Couldn't open file %s\n", filename);
        exit(0);
    }
    list *lines = make_list();
    while((path=fgetl(file))){
        push_list(lines, path);
    }
    fclose(file);
    return lines;
}

data load_image_classification_data(char *images, char *label_file)
{
    list *image_list = get_lines(images);
    list *label_list = get_lines(label_file);
    int k = label_list->size;
    char **labels = (char **)list_to_array(label_list);

    int n = image_list->size;
    node *nd = image_list->front;
    size_t len = 0;
    int i;
    int count = 0;
    tensor x = {0};
    tensor y = tensor_vmake(2, n, k);
    while(nd){
        char *path = (char *)nd->val;
        image im = load_image(path);
        if (!x.size) {
            x = tensor_vmake(4, n, im.c, im.h, im.w);
            len = im.c*im.h*im.w;
        }
        for (i = 0; i < len; ++i){
            x.data[count*len + i] = im.data[i];
        }

        for (i = 0; i < k; ++i){
            if(strstr(path, labels[i])){
                y.data[count*k + i] = 1;
            }
        }
        ++count;
        nd = nd->next;
        free_image(im);
    }

    free_list(image_list);
    free_list(label_list);
    free(labels);

    data d;
    d.x = x;
    d.y = y;
    return d;
}


char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                fprintf(stderr, "malloc failed %ld\n", size);
                exit(0);
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        if(!fgets(&line[curr], readsize, fp)){
            free(line);
            return 0;
        }
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

void free_data(data d)
{
    tensor_free(d.x);
    tensor_free(d.y);
}
