# import os

# folder_path = r"D:\bhanu\tata_paid_demo\combined_model\raw_data_combined\gp_combined_new_v4\train\labels"
# for filename in os.listdir(folder_path):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(folder_path, filename)
#         with open(file_path, 'r') as file:
#             lines = file.readlines()

#         new_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if parts and parts[0] == '1':  # Change class label from 1 to 0
#                 parts[0] = '1'
#             new_lines.append(' '.join(parts))

#         with open(file_path, 'w') as file:
#             file.write('\n'.join(new_lines) + '\n')


import os

folder_path = r"C:\imagevision projects\tata\bottomcut_8oct\train"
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path) as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:  # Ensure the line isn't empty
                parts[0] = "0"  # Change class label to 0
            new_lines.append(" ".join(parts))

        with open(file_path, "w") as file:
            file.write("\n".join(new_lines) + "\n")

print("✅ All class labels updated to 0.")
