# for i in range(10):
#     rule:
#         name: "a-" + str(i) + ".txt"
#         input:
#             f"a-${i}.txt",
#         output:
#             f"b-${i}.txt",
#         shell: "cat {input} > {output}"