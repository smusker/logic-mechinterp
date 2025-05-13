import matplotlib.pyplot as plt

# consistency_proportions = \
#     [
#         [0,0,1,0,0,0,0,0],
#         [0,0,1,1,0,0,0,0],
#         [0,0,0,0,0,0,0,0],
#         [0,0,0,0,0,0,0,0]
#     ]

# rule_name = "hg05"

# component = "block_output"

# token_window_labels = ["746-754", "754-762", "762-770", "770-778",
#                       "778-786", "786-794", "794-802", "802-805"]
# layer_window_labels = ["0-8", "8-16", "16-24", "24-32"]

consistency_proportions = \
    [
        [0,0,0,1,1,0,0,1,1,1],
        [1,1,0,0,1,1,1,1,0,1],
        [0,1,1,1,1,1,1,1,1,1],
        [1,1,1,0,1,1,1,0,1,1],
    ]

rule_name = "hg06"

component = "mlp_output"

token_window_labels = ["728-730", "736-738", "744-746", "752-754", "760-762", "768-770",
                      "776-778", "784-786", "792-794", "800-802"]
layer_window_labels = ["0-3", "8-10", "16-18", "24-26"]

# Plot consistency proportions heatmap
plt.figure(figsize=(12, 8))
plt.imshow(consistency_proportions,
           aspect='auto', cmap='plasma',
           interpolation='nearest')
plt.colorbar(
    label='Categorization Flips'
)
plt.xlabel('Token Windows')
plt.ylabel('Layer Windows')
plt.title(
    f'{rule_name}: Categorization Flips\n'
    f'{component.capitalize()} Component'
)
plt.xticks(
    ticks=range(len(token_window_labels)),
    labels=token_window_labels,
    rotation=90, fontsize=6
)
plt.yticks(ticks=range(len(layer_window_labels)),
           labels=layer_window_labels, fontsize=6)
plt.tight_layout()
# Save the heatmap
filename = f"heatmaps/{rule_name}_{component}_categorization_flip.png"
plt.savefig(filename)
plt.close()