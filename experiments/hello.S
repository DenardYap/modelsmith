.global _start

.section .text
_start:
    mov $1, %rax            # syscall: write
    mov $1, %rdi            # file descriptor: stdout
    mov $message, %rsi      # message to write
    mov $13, %rdx           # message length
    syscall

    mov $60, %rax           # syscall: exit
    xor %rdi, %rdi          # exit code 0
    syscall

.section .data
message:
    .ascii "Hello, world\n"
