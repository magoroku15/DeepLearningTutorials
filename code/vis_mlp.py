"""
This code is used to visualize
http://deeplearning.net/tutorial/mlp.html#tips-and-tricks-for-training-mlps

Usage: put the following on 389th line
    title = "whatever you want"
    plot_pca(classifier, x, train_set_x, train_set_y, index=epoch, title=title)
"""

def plot_pca(classifier, x_symbol, x_data, y_data, index=0,
             title=None, sampling=True):
    import itertools
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(3, 3, figsize=(5, 5))
    axes = axes.flatten()

    apply_hidden = theano.function(inputs=[x_symbol], outputs=classifier.hiddenLayer.output)
    z_data = apply_hidden(x_data.get_value())
    labels = y_data.eval()

    numbers = range(10)
    colors = {0: '#263B1C', 1: '#263374', 2: '#3568B5', 3: '#8A5DDF', 4: '#DBB8EE',
              5: '#46B1C9', 6: '#84C0C6', 7: '#9FB7B9', 8: '#BCC1BA', 9: '#F2E2D2'}

    for ax, prod in zip(axes, zip(numbers[:-1], numbers[1:])):
        # print(ax, prod)
        pca = PCA(n_components=2)
        indexer = numpy.arange(len(labels))[numpy.in1d(labels, prod)]
        label = labels[indexer]
        z = z_data[indexer]
        pca.fit(z)
        z_pca = pca.transform(z)

        if sampling:
           indexer = numpy.arange(len(label))
           numpy.random.shuffle(indexer)
           indexer = indexer[:300]
           z_pca = z_pca[indexer]
           label = label[indexer]

        _c = [colors[l] for l in label]
        ax.scatter(z_pca[:, 0], z_pca[:, 1], color=_c, alpha=0.3)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title('{0}, {1}'.format(prod[0], prod[1]), size='small')
    # plt.show()
    if title is not None:
        fig.suptitle(title)
    plt.savefig('pca_{0:02d}.png'.format(index))