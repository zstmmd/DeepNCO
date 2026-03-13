if __name__ == "__main__":
    maps = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
    digits = "23"
    n = len(digits)
    path = [""] * n
    ans = []


    def dfs(i):
        if i == n:
            ans.append("".join(path.copy()))
            return
        for c in maps[int(digits[i])]:
            path[i] = c
            dfs(i + 1)
            path.pop()


    dfs(0)
