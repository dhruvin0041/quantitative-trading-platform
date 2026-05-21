"use client"

import * as React from "react"
import {
  Moon,
  Sun,
  Laptop,
  Download,
  LayoutTemplate,
  Command as CommandIcon,
} from "lucide-react"

import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command"
import { useTheme } from "next-themes"
import { toast } from "sonner"

export function CommandMenu() {
  const [open, setOpen] = React.useState(false)
  const { setTheme } = useTheme()

  React.useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault()
        setOpen((open) => !open)
      }
    }

    document.addEventListener("keydown", down)
    return () => document.removeEventListener("keydown", down)
  }, [])

  const runCommand = React.useCallback((command: () => void) => {
    setOpen(false)
    command()
  }, [])

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-2 px-3 py-1.5 text-[10px] uppercase tracking-widest font-black text-muted-foreground bg-white/5 border border-white/10 rounded-md hover:bg-white/10 hover:text-foreground transition-all"
      >
        <CommandIcon className="w-3 h-3" />
        <span className="hidden sm:inline">Terminal Command</span>
        <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-white/10 bg-black/40 px-1.5 font-mono text-[10px] font-medium text-muted-foreground opacity-100">
          <span className="text-xs">⌘</span>K
        </kbd>
      </button>
      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Execute command..." className="font-mono" />
        <CommandList className="font-sans border-t border-white/5 bg-black/90 backdrop-blur-xl">
          <CommandEmpty className="py-6 text-center text-xs font-mono opacity-50 uppercase tracking-widest">No matching command sequence.</CommandEmpty>
          <CommandGroup heading="Intelligence Actions">
            <CommandItem onSelect={() => runCommand(() => toast.info("Generating performance report..."))} className="py-3">
              <Download className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium">Export Terminal State</span>
              <CommandShortcut className="font-mono text-[10px]">⌘E</CommandShortcut>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => toast.success("Terminal layout synchronized"))} className="py-3">
              <LayoutTemplate className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium">Synchronize Layout</span>
              <CommandShortcut className="font-mono text-[10px]">⌘S</CommandShortcut>
            </CommandItem>
          </CommandGroup>
          <CommandSeparator className="bg-white/5" />
          <CommandGroup heading="System Protocols">
            <CommandItem onSelect={() => runCommand(() => setTheme("light"))} className="py-3">
              <Sun className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium">Switch to Light Protocol</span>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => setTheme("dark"))} className="py-3">
              <Moon className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium">Switch to Dark Protocol (OLED)</span>
            </CommandItem>
            <CommandItem onSelect={() => runCommand(() => setTheme("system"))} className="py-3">
              <Laptop className="mr-3 h-4 w-4 opacity-70" />
              <span className="text-sm font-medium">Follow System OS</span>
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  )
}

